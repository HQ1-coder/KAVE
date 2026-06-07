import torch


_GRAY_SYMBOL_BITS = torch.tensor(
    [[0, 0], [0, 1], [1, 1], [1, 0]],
    dtype=torch.long)
_GRAY_BITS_TO_SYMBOL = torch.tensor([0, 1, 3, 2], dtype=torch.long)


def expand_gray_symbols(symbols, k=4):
    if symbols.numel() == 0:
        return torch.empty(0, device=symbols.device, dtype=torch.long)
    if k == 2:
        return symbols.to(torch.long).unsqueeze(-1).reshape(-1)
    return _GRAY_SYMBOL_BITS.to(symbols.device)[symbols].reshape(-1)


def gray_bits_to_symbols(bits, k=4):
    if bits.numel() == 0:
        return torch.empty(0, device=bits.device, dtype=torch.long)
    if k == 2:
        return bits.to(torch.long).reshape(-1)
    if bits.numel() % 2 != 0:
        raise ValueError("Gray bitstream length must be even.")
    bit_pairs = bits.reshape(-1, 2).to(torch.long)
    pair_indices = bit_pairs[:, 0] * 2 + bit_pairs[:, 1]
    return _GRAY_BITS_TO_SYMBOL.to(bits.device)[pair_indices]


def usable_channel_symbol_count(transmitted_symbol_count, channel_coding):
    transmitted_symbol_count = max(int(transmitted_symbol_count), 0)
    if channel_coding == "repetition":
        return transmitted_symbol_count
    if channel_coding == "hamming84":
        return (transmitted_symbol_count // 4) * 4
    raise ValueError(f"Unsupported channel_coding: {channel_coding}")


def hamming84_encode(info_bits):
    if info_bits.numel() == 0:
        return torch.empty(info_bits.shape[0], 8, device=info_bits.device, dtype=torch.long)
    if info_bits.shape[-1] != 4:
        raise ValueError("Hamming84 expects 4 information bits per codeword.")

    d0, d1, d2, d3 = info_bits.to(torch.long).unbind(dim=1)
    p1 = d0 ^ d1 ^ d3
    p2 = d0 ^ d2 ^ d3
    p4 = d1 ^ d2 ^ d3
    p8 = p1 ^ p2 ^ d0 ^ p4 ^ d1 ^ d2 ^ d3
    return torch.stack([p1, p2, d0, p4, d1, d2, d3, p8], dim=1)


def _build_hamming84_codebook(device):
    payload_indices = torch.arange(16, device=device, dtype=torch.long)
    bit_shifts = torch.tensor([3, 2, 1, 0], device=device, dtype=torch.long)
    payload_bits = ((payload_indices.unsqueeze(1) >> bit_shifts) & 1).to(torch.long)
    code_bits = hamming84_encode(payload_bits)
    transmit_symbols = gray_bits_to_symbols(code_bits.reshape(-1)).view(-1, 4)
    return payload_bits, transmit_symbols


def prepare_channel_message(transmitted_symbol_count, seed, device, channel_coding, k=4):
    transmitted_symbol_count = usable_channel_symbol_count(
        transmitted_symbol_count, channel_coding)
    empty_symbols = torch.empty(0, device=device, dtype=torch.long)
    empty_bits = torch.empty(0, device=device, dtype=torch.long)
    if transmitted_symbol_count == 0:
        return empty_symbols, empty_bits, 0

    message_generator = torch.Generator(device=str(device))
    message_generator.manual_seed(seed + 1)

    if channel_coding == "repetition":
        transmit_symbols = torch.randint(
            0,
            k,
            (transmitted_symbol_count,),
            device=device,
            generator=message_generator)
        reference_bits = expand_gray_symbols(transmit_symbols, k=k)
        bits_per_symbol = 1 if k == 2 else 2
        payload_symbol_count = reference_bits.numel() // bits_per_symbol
        return transmit_symbols, reference_bits, payload_symbol_count

    if channel_coding == "hamming84":
        if k != 4:
            raise ValueError("hamming84 currently supports K=4 only")
        codeword_count = transmitted_symbol_count // 4
        info_bits = torch.randint(
            0,
            2,
            (codeword_count, 4),
            device=device,
            generator=message_generator)
        code_bits = hamming84_encode(info_bits)
        transmit_symbols = gray_bits_to_symbols(code_bits.reshape(-1), k=k)
        reference_bits = info_bits.reshape(-1)
        payload_symbol_count = reference_bits.numel() // 2
        return transmit_symbols, reference_bits, payload_symbol_count

    raise ValueError(f"Unsupported channel_coding: {channel_coding}")


def aggregate_channel_scores(pair_scores,
                             symbol_group_ids,
                             symbol_group_sizes,
                             channel_coding,
                             repetition_decoder_mode="hard_sum",
                             repetition_gap_scale=1.0):
    grouped_scores = torch.zeros(
        pair_scores.shape[0],
        symbol_group_sizes.numel(),
        device=pair_scores.device,
        dtype=pair_scores.dtype)
    if pair_scores.numel() == 0 or symbol_group_sizes.numel() == 0:
        return grouped_scores, None, None

    pair_weights = None
    grouped_bit_llrs = None
    weighted_pair_scores = pair_scores

    if channel_coding == "repetition":
        if repetition_decoder_mode not in {"hard_sum", "confidence_weighted", "gray_llr"}:
            raise ValueError(
                f"Unsupported repetition_decoder_mode: {repetition_decoder_mode}")
        if repetition_gap_scale <= 0.0:
            raise ValueError("repetition_gap_scale must be > 0")
        if repetition_decoder_mode == "confidence_weighted":
            pair_confidence = torch.topk(
                pair_scores,
                k=2,
                largest=True,
                dim=0).values
            pair_confidence = pair_confidence[0] - pair_confidence[1]
            pair_weights = torch.empty_like(pair_confidence)
            start = 0
            for group_size in symbol_group_sizes.tolist():
                end = start + int(group_size)
                group_confidence = pair_confidence[start:end]
                group_weights = torch.softmax(
                    group_confidence * repetition_gap_scale,
                    dim=0) * float(group_size)
                pair_weights[start:end] = group_weights
                start = end
            weighted_pair_scores = pair_scores * pair_weights.unsqueeze(0)
        elif repetition_decoder_mode == "gray_llr":
            scaled_pair_scores = pair_scores * repetition_gap_scale
            pair_bit_llrs = []
            gray_symbol_bits = _GRAY_SYMBOL_BITS.to(pair_scores.device).bool()
            for bit_idx in range(gray_symbol_bits.shape[1]):
                bit_mask = gray_symbol_bits[:, bit_idx]
                bit1_scores = torch.logsumexp(scaled_pair_scores[bit_mask], dim=0)
                bit0_scores = torch.logsumexp(scaled_pair_scores[~bit_mask], dim=0)
                pair_bit_llrs.append(bit1_scores - bit0_scores)
            pair_bit_llrs = torch.stack(pair_bit_llrs, dim=0)
            grouped_bit_llrs = torch.zeros(
                pair_bit_llrs.shape[0],
                symbol_group_sizes.numel(),
                device=pair_scores.device,
                dtype=pair_scores.dtype)
            bit_group_index = symbol_group_ids.unsqueeze(0).expand(
                pair_bit_llrs.shape[0],
                -1)
            grouped_bit_llrs.scatter_add_(1, bit_group_index, pair_bit_llrs)

    group_index = symbol_group_ids.unsqueeze(0).expand(pair_scores.shape[0], -1)
    grouped_scores.scatter_add_(1, group_index, weighted_pair_scores)
    return grouped_scores, pair_weights, grouped_bit_llrs


def decode_channel_scores(grouped_scores,
                          channel_coding,
                          repetition_decoder_mode="hard_sum",
                          grouped_bit_llrs=None,
                          k=4):
    symbol_count = usable_channel_symbol_count(grouped_scores.shape[1], channel_coding)
    grouped_scores = grouped_scores[:, :symbol_count]
    empty_bits = torch.empty(0, device=grouped_scores.device, dtype=torch.long)
    empty_gaps = torch.empty(0, device=grouped_scores.device, dtype=grouped_scores.dtype)
    bits_per_symbol = 1 if k == 2 else 2

    if symbol_count == 0:
        return empty_bits, 0, empty_gaps

    if channel_coding == "repetition":
        if repetition_decoder_mode == "gray_llr":
            if k == 2:
                top2_scores = torch.topk(grouped_scores, k=2, largest=True, dim=0).values
                decision_gaps = top2_scores[0] - top2_scores[1]
                recovered_symbols = grouped_scores.argmax(dim=0)
                recovered_bits = expand_gray_symbols(recovered_symbols.to(torch.long), k=2)
                payload_symbol_count = recovered_bits.numel() // bits_per_symbol
                return recovered_bits, payload_symbol_count, decision_gaps
            if grouped_bit_llrs is None:
                raise ValueError("grouped_bit_llrs is required for gray_llr decoding")
            grouped_bit_llrs = grouped_bit_llrs[:, :symbol_count]
            symbol_bit_llrs = grouped_bit_llrs.transpose(0, 1)
            decision_gaps = symbol_bit_llrs.abs().min(dim=1).values
            recovered_bits = (symbol_bit_llrs > 0).to(torch.long).reshape(-1)
            payload_symbol_count = recovered_bits.numel() // 2
            return recovered_bits, payload_symbol_count, decision_gaps

        top2_scores = torch.topk(grouped_scores, k=2, largest=True, dim=0).values
        decision_gaps = top2_scores[0] - top2_scores[1]
        recovered_symbols = grouped_scores.argmax(dim=0)
        recovered_bits = expand_gray_symbols(recovered_symbols.to(torch.long), k=k)
        payload_symbol_count = recovered_bits.numel() // bits_per_symbol
        return recovered_bits, payload_symbol_count, decision_gaps

    if channel_coding == "hamming84":
        if k != 4:
            raise ValueError("hamming84 currently supports K=4 only")
        codeword_count = symbol_count // 4
        if codeword_count == 0:
            return empty_bits, 0, empty_gaps

        payload_codebook, symbol_codebook = _build_hamming84_codebook(grouped_scores.device)
        per_symbol_scores = grouped_scores.transpose(0, 1).reshape(codeword_count, 4, 4)
        codeword_scores = torch.zeros(
            codeword_count,
            payload_codebook.shape[0],
            device=grouped_scores.device,
            dtype=grouped_scores.dtype)
        for symbol_pos in range(4):
            codeword_scores += per_symbol_scores[:, symbol_pos, :][:, symbol_codebook[:, symbol_pos]]

        top2_scores = torch.topk(codeword_scores, k=2, largest=True, dim=1).values
        decision_gaps = top2_scores[:, 0] - top2_scores[:, 1]
        recovered_payload_indices = codeword_scores.argmax(dim=1)
        recovered_bits = payload_codebook[recovered_payload_indices].reshape(-1)
        payload_symbol_count = recovered_bits.numel() // 2
        return recovered_bits, payload_symbol_count, decision_gaps

    raise ValueError(f"Unsupported channel_coding: {channel_coding}")