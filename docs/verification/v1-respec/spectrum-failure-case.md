# One failed Spectrum Peak episode

Selection was fixed before reconstructing audio: the first failed `spectrum-peak`
record under `noise-10db`, in the frozen robustness report's order. This is a
single-case explanation, not a prevalence estimate or a new evaluation cohort.

Episode `benchmark-iid-0135` has source **6857 cents**, target **6100 cents**, and
nuisance seed `14708078973187757233`. The reconstructed initial waveform and
public-spectrum SHA-256 hashes exactly match the original result. Its realized
SNR is 10.0000000006 dB. No model was loaded, trained, or adapted.

| Measurement on the same noisy waveform | Estimated pitch | Error from source |
| --- | ---: | ---: |
| Maximum linear Hann-FFT bin | 6856.972712 cents | -0.027288 cents |
| Existing `analyze()` quadratic FFT interpolation | 6857.001887 cents | +0.001887 cents |
| Public 1961-bin spectrum, feasible argmax | 5935 cents | -922 cents |

The true frequency is 429.205984 Hz. The 262,144-point FFT has 0.183105 Hz bin
spacing and a Hann main-lobe first-null offset of approximately 0.366211 Hz.
The public log-grid's adjacent points are 428.710432 Hz (6855 cents) and
429.950386 Hz (6860 cents). Their offsets from truth are -0.495552 and +0.744402 Hz:
both fall outside the main lobe. The encoder point-interpolates calibrated dB
levels at these grid frequencies; it does not preserve a frequency band's peak.

The full FFT's peak is **-18.009 dBFS**. The adjacent public-grid values are only
**-71.384** and **-74.644 dBFS**. At the winning incorrect coordinate, 5935 cents,
the sampled noisy level is **-70.174 dBFS**. The isolated added-noise component
produces the same winning value within 0.000001 dB; the clean component there is
below the -120 dBFS analysis floor. On the same source's clean capture the public
grid correctly selects 6855 cents, so noise overtakes the attenuated sampled
signal in this case.

The controller then applies the +165-cent plan implied by estimate 5935 and
target 6100. The true final pitch is 7022 cents, matching the stored **+922-cent
submitted failure**, with zero invalid actions. This particular failure originates
in the initial public-spectrum estimate rather than an incorrect execution of
the committed plan.

This supports the proposed narrow-peak undersampling mechanism for this observed
failure. It does not establish that all failures share that cause, measure the
performance of a different representation, or justify a post-hoc encoder change
inside the frozen qualification protocol.

Reproduction: copy `diagnose-spectrum-case.py` and the hash-identified complete
`robustness.json` from `outputs/v1-respec-qualification/` into a new scratch
directory, then run the copied script with this revision's source on `PYTHONPATH`,
NumPy available, and one CPU thread. The script
asserts exact waveform/spectrum hashes and exact public-spectrum reconstruction,
then writes `spectrum-failure-case.json`, the 481 feasible grid samples in
`spectrum-failure-case-grid.csv`, and FFT samples around truth in
`spectrum-failure-case-linear.csv`.

Input robustness report SHA-256:
`8c988346cfa712d692502305fef2dfab953806ccc7f07b3b583153c31575b831`.
