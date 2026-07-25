# Mathematical Formulation — CAILP-Social

## Problem

Given attributed undirected social graph \(G=(V,E,X,Y)\) and retention budget \(\rho\in(0,1]\), learn edge importance \(s_e\in[0,1]\) and return \(E_\rho\subset E\) with \(|E_\rho|\approx\rho|E|\) maximizing predictive/structural/group utility.

## Inverse Link Prediction

Traditional LP: \(P((i,j)\notin E)\).  
ILP (extension of Bangian Tabrizi et al., 2025): score existing \(e\in E\) by deletion harm.

## Encoders

\[
H=f_\theta(X,A),\quad h_i\in\mathbb{R}^d
\]

\[
z_{ij}^{\mathrm{node}}=[h_i+h_j\,\|\,|h_i-h_j|\,\|\,h_i\odot h_j\,\|\,\cos(h_i,h_j)\,\|\,\phi(i,j)]
\]

\[
z_{ij}^{\mathrm{edge}}=g_\psi(q_{ij},L(G))\ \text{or local encoder}
\]

## Counterfactual target

\[
y_e^{\mathrm{cf}}=\mathrm{Normalize}\Big(\sum_{k\in\{\mathrm{task,comm,conn,spec,repr,group}\}}\beta_k\Delta_k(e)\Big)
\]

A31: \(y_e^{\mathrm{cf,1}}=\mathrm{Normalize}(\Delta_{\mathrm{task}})\) only.

## Fusion & decoder

Cross-attention (one of several ablated fusions):

\[
u=\mathrm{Attn}(W_Q z^{\mathrm{node}},W_K z^{\mathrm{edge}},W_V z^{\mathrm{edge}}),\quad
z=\mathrm{LN}(W_n z^{\mathrm{node}}+W_u u)
\]

\[
(\mu_e,\log\sigma_e^2)=\mathrm{Dec}(z),\quad s_e=\mu_e
\]

## Loss

\[
\mathcal{L}=\lambda_{\mathrm{task}}\mathcal{L}_{\mathrm{CE}}+\lambda_{\mathrm{cf}}\mathcal{L}_{\mathrm{NLL}}(\mu,\sigma;y^{\mathrm{cf}})+\lambda_{\mathrm{rank}}\mathcal{L}_{\mathrm{hinge}}+\cdots
\]

## Constrained pruning

Rank ascending by \(s_e\); delete iff constraints satisfied until budget; report shortfalls.

## Hypotheses (incl. RQ11)

See research brief. Multi-objective aggregation retained as headline only if it beats A31 on ≥2 metrics × ≥2 datasets with fusion held fixed.
