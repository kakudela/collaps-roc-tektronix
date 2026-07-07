# muon-pmt-calibration

Cosmic muon calibration setup: a Tektronix MSO46 oscilloscope connected over
USB to a small inner (trigger) detector PMT and 4 outer scintillator PMTs.

- **CH1** = trigger PMT (inner detector), edge trigger, falling slope
- **CH2-CH5** = outer scintillator PMTs

## Hardware / workflow overview

```
 [scope + PMTs] --USB--> [this laptop]  --rsync-->  [submit.mit.edu]
                          take_data.py                analyze_run.py
                          (acquisition)                (RDataFrame analysis,
                                                         writes to public_html)
```

The scope only has a physical USB connection to the laptop it's plugged into,
so `take_data.py` only ever runs there. `analyze_run.py` only needs PyROOT, so
it runs on `submit` (or anywhere with ROOT installed) against data copied over
from the laptop.

## 1. Acquiring data (on the laptop plugged into the scope)

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python take_data.py --frames 200 --trigger-level -0.01
```

This creates `data/run_YYYYMMDD_HHMMSS/`, containing:
- `metadata.json` -- every scope setting (channel scale/termination/bandwidth,
  trigger config, sample rate) plus the raw-ADC-to-volts/time conversion
  factors (`ymult`/`yzero`/`yoff`/`xincr`/`xzero`) for each channel.
- `batchNNNNN.root` -- one ROOT file per 200-event batch. Written once,
  never reopened, so a crash mid-run only ever loses the batch in progress.
  Branches: `ch1_raw`...`ch5_raw` (raw int16 ADC waveforms), `frame`,
  `batch_start_unix`, `duration_s`.

Stop with Ctrl+C at any point -- already-written batches stay valid.

## 2. Getting data to submit

Raw data and analysis output both live under `public_html` so they're easy
to browse/share:

```bash
rsync -avz data/run_YYYYMMDD_HHMMSS/ \
    kudela@submit.mit.edu:~/public_html/muon_pmt_calibration/run_YYYYMMDD_HHMMSS/
```

## 3. Analyzing (on submit)

```bash
ssh kudela@submit.mit.edu
cd ~/collaps-roc-tektronix/muon-pmt-calibration
python3 analyze_run.py ~/public_html/muon_pmt_calibration/run_YYYYMMDD_HHMMSS
```

`analyze_run.py` defaults to writing its output (`analysis.root` + PNGs) into
`<run_dir>/analysis/` -- since `run_dir` is already under `public_html`, the
analysis output ends up there automatically too, browsable at
`https://submit.mit.edu/~kudela/muon_pmt_calibration/run_YYYYMMDD_HHMMSS/analysis/`.

Per channel, it computes (see the C++ block at the top of the script for the
exact math):
- `baseline` -- mean ADC value in the first N samples, before the pulse
- `integral_pC` -- baseline-subtracted pulse charge (assumes 50-ohm
  termination), the main calibration quantity
- `peak_mv` -- pulse depth in mV
- `dt_samples` -- timing offset of each outer PMT's peak relative to CH1,
  used to separate genuine coincidences from uncorrelated background

Bin ranges in the script are rough starting guesses -- rerun with `--outdir`
and rebin once you've looked at the real histograms for your setup.
