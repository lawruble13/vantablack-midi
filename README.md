# vantablack-midi

Current best results obtoined using `windowedApproximation.m`, depends on `linearApproximation.m`.
This branch attempts to parallelize the algorithm used in the main branch by computing the best possible approximation for many sequential segments (which are back-to-back) independently, then compute patches which attempt to smooth the transition between sequential segments  independently (patches are limited so that regardless of patch width, a note played by one patch will not affect the next patch).
