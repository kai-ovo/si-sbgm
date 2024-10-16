# Stochastic Interpolants v.s. Score-Based Generative Models
This repo contains a personal exploration on comparing Stochastic Interpolants (SI) [1,2]and Score-Based Generative Models (SBGM) [3]. The comparisons are carried out using Entropic Optimal Transport, in particular the Sinkhorn's algorithm [4]. `report.pdf` contains the results and a SIAM-style self-contained report. Part of the SI implementations are adapted from [here](https://github.com/malbergo/stochastic-interpolants); part of the SBGM implementations are adapted from [here](https://github.com/yang-song/score_sde). The Sinkhorn's algorithm is a personal implementation, which works very well, but improvements might be possible.


---

## References
[1] Albergo, Michael Samuel, and Eric Vanden-Eijnden. "Building Normalizing Flows with Stochastic Interpolants." The Eleventh International Conference on Learning Representations.

[2] Albergo, Michael S., Nicholas M. Boffi, and Eric Vanden-Eijnden. "Stochastic interpolants: A unifying framework for flows and diffusions." arXiv preprint arXiv:2303.08797 (2023).

[3] Song, Yang, et al. "Score-Based Generative Modeling through Stochastic Differential Equations." International Conference on Learning Representations.

[4] Cuturi, Marco. "Sinkhorn distances: Lightspeed computation of optimal transport." Advances in neural information processing systems 26 (2013).