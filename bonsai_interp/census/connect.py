"""Weight connection matrix between MLP layers, through the residual stream only (attention and intermediate norms are
ignored).

For a writer neuron A (output column v_A) and a reader neuron B in a later layer with MLP-input norm gain ĝ_B and typical
input RMS r_B (the residual before B's MLP on the lens contexts):
    G[B, A] = W_gate_B · (ĝ_B ⊙ v_A) / r_B,   U[B, A] = W_up_B · (ĝ_B ⊙ v_A) / r_B,   P = G · U.
One unit of A's activation moves B's gate pre-activation by G and its up projection by U; P is the second-order term of
B's product that A drives alone. P does not depend on A's sign (both A poles drive B the same way), and its sign says
which pole of B it pushes. G and U are kept so a reader can form the first-order, sign-aware effect at an operating
point: Δact_B ≈ (silu'(g0)·u0·G + silu(g0)·U)·a_A.
Kept per neuron: the K readers (later neurons) with the largest |P|; per pole of B: the K writers with the largest
P (+ pole) or −P (− pole). Links are encoded as layer * N + neuron.
"""
import numpy as np
import torch

K_LINK = 24


def _merge(val, idx, G, U, nv, ni, nG, nU, k):
    v = torch.cat([val, nv], 1)
    top, j = v.topk(k, dim=1)
    take = lambda a, b: torch.cat([a, b], 1).gather(1, j)
    return top, take(idx, ni), take(G, nG), take(U, nU)


def links(model, rB, tick=lambda: None, k=K_LINK, chunk=1024):
    layers = model.model.layers
    nL = len(layers)
    Wd0 = layers[0].mlp.down_proj
    d, N = Wd0.out_features, Wd0.in_features
    dev = model.lm_head.weight.device
    blank = lambda: [torch.full((N, k), -float("inf"), device=dev), torch.full((N, k), -1, dtype=torch.int32, device=dev),
                     torch.zeros(N, k, device=dev), torch.zeros(N, k, device=dev)]
    rd = [blank() for _ in range(nL)]            # readers of A
    wr = [[blank(), blank()] for _ in range(nL)]  # writers of B, + and − pole
    for B in range(1, nL):
        gB = layers[B].post_attention_layernorm.weight.float() / float(rB[B])
        Wg = (layers[B].mlp.gate_proj.weight().float() * gB).to(torch.bfloat16)  # (N, d)
        Wu = (layers[B].mlp.up_proj.weight().float() * gB).to(torch.bfloat16)
        for A in range(B):
            tick()
            WdA = layers[A].mlp.down_proj.weight()  # (d, N) bf16
            for c0 in range(0, N, chunk):
                cols = WdA[:, c0:c0 + chunk]
                G = (Wg @ cols).float()  # (N_B, ch)
                Uc = (Wu @ cols).float()
                P = G * Uc
                ch = P.shape[1]
                # writers of B (rows), over A's neurons in this chunk
                for p, s in ((0, 1.0), (1, -1.0)):
                    v, j = (s * P).topk(k, dim=1)
                    wr[B][p] = list(_merge(*wr[B][p], v, (j + A * N + c0).int(), G.gather(1, j), Uc.gather(1, j), k))
                # readers of A (columns), over B's neurons
                v, j = P.abs().topk(k, dim=0)  # (k, ch)
                cur = [t[c0:c0 + ch] for t in rd[A]]
                new = _merge(*cur, v.T, (j.T + B * N).int(), G.gather(0, j).T, Uc.gather(0, j).T, k)
                for t, nt in zip(rd[A], new):
                    t[c0:c0 + ch] = nt
                del G, Uc, P
            del WdA
        del Wg, Wu
        print(f"census links: reader layer {B} done", flush=True)
    to = lambda s: {"idx": s[1].cpu().numpy(), "G": s[2].cpu().numpy(), "U": s[3].cpu().numpy()}
    return [{"rd": to(rd[L]), "wr+": to(wr[L][0]), "wr-": to(wr[L][1])} for L in range(nL)]
