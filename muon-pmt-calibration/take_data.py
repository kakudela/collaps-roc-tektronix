#!/usr/bin/env python3
"""
Data acquisition script for Tektronix MSO46, connected via USB.

Setup: CH1 = trigger PMT (inner detector).
       CH2-CH5 = the 4 PMTs on the large outer scintillator.

Each run acquires a batch of triggered events using FastFrame (so the scope
captures many triggers back-to-back with minimal dead time), then reads out
all 5 channels for every frame in that batch and saves them as a ROOT file
(one file per batch, so a crash never corrupts previously-saved batches).
Each run gets its own folder under data/, containing metadata.json plus one
batch###.root per completed batch, ready to open directly with RDataFrame
(e.g. ROOT::RDataFrame("events", "data/run_.../batch*.root")).
"""
import argparse
import json
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyvisa
import uproot

TRIGGER_CHANNEL = 1
PMT_CHANNELS = [2, 3, 4, 5]
ALL_CHANNELS = [TRIGGER_CHANNEL] + PMT_CHANNELS

# Tune these to your actual signal amplitudes before a real run, these are
# just placeholders.
CHANNEL_SETTINGS = {
    # CH1 at 100mV/div flat out stopped triggering (tested live, 0 triggers
    # in 10+ sec at what should be ~2Hz). 20mV/div works, but it means CH1's
    # own waveform clips in the data. Fine since CH1 is only used for
    # triggering/timing, we never look at its pulse height.
    1: dict(scale=0.02, offset=0.0, termination=50.0),   # trigger PMT (inner detector)
    2: dict(scale=0.1,  offset=0.0, termination=50.0),   # outer PMT 1
    3: dict(scale=0.1,  offset=0.0, termination=50.0),   # outer PMT 2
    4: dict(scale=0.1,  offset=0.0, termination=50.0),   # outer PMT 3
    5: dict(scale=0.1,  offset=0.0, termination=50.0),   # outer PMT 4
}

VISA_RESOURCE = "USB0::0x0699::0x0527::C031835::0::INSTR"


def find_scope(rm):
    resources = rm.list_resources()
    matches = [r for r in resources if "0x0699" in r.upper() or "1689" in r]
    if not matches:
        sys.exit(f"No MSO46 found on USB. Resources seen: {resources}")
    return matches[0]


def check_errors(inst, context=""):
    msg = inst.query("EVMSG?").strip()
    code = msg.split(",", 1)[0]
    if code not in ("0", "1"):
        print(f"[scope error] {context}: {msg}")


def configure_scope(inst, trigger_level, record_length, fastframe_count):
    inst.write("*CLS")
    inst.write("ACQ:STATE STOP")
    inst.write("HORIZONTAL:MODE MANUAL")
    inst.write(f"HORIZONTAL:RECORDLENGTH {record_length}")

    for ch, s in CHANNEL_SETTINGS.items():
        inst.write(f"SELECT:CH{ch} ON")
        inst.write(f"CH{ch}:TERMINATION {s['termination']}")
        inst.write(f"CH{ch}:SCALE {s['scale']}")
        inst.write(f"CH{ch}:OFFSET {s['offset']}")
        inst.write(f"CH{ch}:POSITION 0")  # don't inherit stray front-panel offsets

    inst.write("TRIGGER:A:MODE NORMAL")
    inst.write("TRIGGER:A:TYPE EDGE")
    inst.write(f"TRIGGER:A:EDGE:SOURCE CH{TRIGGER_CHANNEL}")
    inst.write("TRIGGER:A:EDGE:SLOPE FALL")  # flip to RISE if your pulses are positive-going
    inst.write(f"TRIGGER:A:LEVEL:CH{TRIGGER_CHANNEL} {trigger_level}")

    inst.write("HORIZONTAL:FASTFRAME:STATE ON")
    inst.write(f"HORIZONTAL:FASTFRAME:COUNT {fastframe_count}")

    inst.write("DATA:START 1")
    inst.write(f"DATA:STOP {record_length}")
    inst.write("DATA:ENCDG RIBINARY")
    inst.write("DATA:WIDTH 2")
    check_errors(inst, "configure_scope")


def get_channel_hw_settings(inst, ch):
    """Front-panel-visible settings for a channel: V/div, termination, BW limit, etc."""
    return dict(
        scale_v_div=float(inst.query(f"CH{ch}:SCALE?")),
        offset_v=float(inst.query(f"CH{ch}:OFFSET?")),
        position_div=float(inst.query(f"CH{ch}:POSITION?")),
        termination_ohm=float(inst.query(f"CH{ch}:TERMINATION?")),
        bandwidth_hz=float(inst.query(f"CH{ch}:BANDWIDTH?")),
        coupling=inst.query(f"CH{ch}:COUPLING?").strip(),
        invert=bool(int(inst.query(f"CH{ch}:INVERT?"))),
        probe_gain=float(inst.query(f"CH{ch}:PROBE:GAIN?")),
    )


def get_waveform_scaling(inst, ch):
    """Only valid once the scope actually holds acquired data for this channel."""
    inst.write(f"DATA:SOURCE CH{ch}")
    time.sleep(0.05)
    return dict(
        ymult=float(inst.query("WFMOUTPRE:YMULT?")),
        yzero=float(inst.query("WFMOUTPRE:YZERO?")),
        yoff=float(inst.query("WFMOUTPRE:YOFF?")),
        xincr=float(inst.query("WFMOUTPRE:XINCR?")),
        xzero=float(inst.query("WFMOUTPRE:XZERO?")),
        nr_pt=int(inst.query("WFMOUTPRE:NR_PT?")),
        byt_or=inst.query("WFMOUTPRE:BYT_OR?").strip(),
        bn_fmt=inst.query("WFMOUTPRE:BN_FMT?").strip(),
    )


def get_full_metadata(inst, include_waveform_scaling):
    """Everything you'd need to interpret the raw ADC counts later: channel
    settings, trigger config, horizontal settings, and (once there's real
    data) the WFMOUTPRE scaling per channel."""
    meta = {
        "idn": inst.query("*IDN?").strip(),
        "captured_at": datetime.now().isoformat(),
        "acquisition_mode": inst.query("ACQUIRE:MODE?").strip(),
        "record_length": int(inst.query("HORIZONTAL:RECORDLENGTH?")),
        "sample_rate_hz": float(inst.query("HORIZONTAL:SAMPLERATE?")),
        "fastframe_count": int(inst.query("HORIZONTAL:FASTFRAME:COUNT?")),
        "trigger": {
            "mode": inst.query("TRIGGER:A:MODE?").strip(),
            "type": inst.query("TRIGGER:A:TYPE?").strip(),
            "source": f"CH{TRIGGER_CHANNEL}",
            "slope": inst.query("TRIGGER:A:EDGE:SLOPE?").strip(),
            "level_v": float(inst.query(f"TRIGGER:A:LEVEL:CH{TRIGGER_CHANNEL}?")),
            "coupling": inst.query("TRIGGER:A:EDGE:COUPLING?").strip(),
        },
        "channels": {},
    }
    for ch in ALL_CHANNELS:
        chmeta = get_channel_hw_settings(inst, ch)
        if include_waveform_scaling:
            chmeta.update(get_waveform_scaling(inst, ch))
        meta["channels"][str(ch)] = chmeta
    return meta


def parse_ieee_blocks(raw, n_blocks):
    """Parse `n_blocks` '#<ndigits><nbytes><data>' IEEE488.2 blocks, separated by ';'."""
    blocks = []
    pos = 0
    for i in range(n_blocks):
        if i > 0:
            assert raw[pos:pos + 1] == b";", f"expected ';' separator at {pos}: {raw[pos:pos+10]!r}"
            pos += 1
        assert raw[pos:pos + 1] == b"#", f"bad block header at {pos}: {raw[pos:pos+10]!r}"
        ndigits = int(raw[pos + 1:pos + 2])
        nbytes = int(raw[pos + 2:pos + 2 + ndigits])
        start = pos + 2 + ndigits
        blocks.append(raw[start:start + nbytes])
        pos = start + nbytes
    return blocks


def acquire_batch(inst, n_frames, record_length):
    inst.write("ACQ:STOPAFTER SEQUENCE")
    inst.write("ACQ:STATE RUN")

    t0 = time.time()
    while True:
        state = inst.query("ACQ:STATE?").strip()
        if state == "0":
            break
        if time.time() - t0 > max(30, n_frames * 0.5):
            sys.exit("Timed out waiting for FastFrame acquisition to complete "
                      "(no triggers seen -- check trigger level/source/cabling).")
        time.sleep(0.1)

    source_list = ",".join(f"CH{ch}" for ch in ALL_CHANNELS)
    waveforms = np.empty((n_frames, len(ALL_CHANNELS), record_length), dtype=np.int16)

    for frame in range(1, n_frames + 1):
        inst.write(f"DATA:FRAMESTART {frame}")
        inst.write(f"DATA:FRAMESTOP {frame}")
        inst.write(f"DATA:SOURCE {source_list}")
        inst.write("CURVE?")
        raw = inst.read_raw()
        blocks = parse_ieee_blocks(raw, len(ALL_CHANNELS))
        for i, block in enumerate(blocks):
            waveforms[frame - 1, i, :] = np.frombuffer(block, dtype=">i2")

    return waveforms


def save_batch_root(path, waveforms, batch_start_iso, duration_s):
    """One ROOT file per batch: open, fill, close, done. Never reopened, so
    a crash only ever costs you the batch in progress, everything already
    written to disk stays valid."""
    n_events = waveforms.shape[0]
    branch_types = {"frame": "int32", "batch_start_unix": "float64", "duration_s": "float64"}
    branch_types.update({f"ch{ch}_raw": ("int16", (waveforms.shape[2],)) for ch in ALL_CHANNELS})

    batch_start_unix = datetime.fromisoformat(batch_start_iso).timestamp()
    data = {
        "frame": np.arange(n_events, dtype=np.int32),
        "batch_start_unix": np.full(n_events, batch_start_unix, dtype=np.float64),
        "duration_s": np.full(n_events, duration_s, dtype=np.float64),
    }
    for i, ch in enumerate(ALL_CHANNELS):
        data[f"ch{ch}_raw"] = waveforms[:, i, :]

    with uproot.recreate(path) as f:
        f.mktree("events", branch_types)
        f["events"].extend(data)


def main():
    ap = argparse.ArgumentParser(description="Take FastFrame data from Tektronix MSO46 over USB")
    ap.add_argument("--frames", type=int, default=200, help="frames (events) per batch")
    ap.add_argument("--batches", type=int, default=0, help="number of batches to take (0 = run until Ctrl+C)")
    ap.add_argument("--record-length", type=int, default=1250)
    ap.add_argument("--trigger-level", type=float, default=-0.01, help="volts, on CH1")
    ap.add_argument("--output", type=str, default="data")
    args = ap.parse_args()

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    outdir = Path(args.output) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    metadata_path = outdir / "metadata.json"

    rm = pyvisa.ResourceManager("@py")
    resource = find_scope(rm)
    print(f"Connecting to {resource} ...")
    inst = rm.open_resource(resource)
    inst.timeout = 10000
    print("IDN:", inst.query("*IDN?").strip())

    configure_scope(inst, args.trigger_level, args.record_length, args.frames)

    metadata = get_full_metadata(inst, include_waveform_scaling=False)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Run folder: {outdir}")

    batch = 0
    try:
        while args.batches == 0 or batch < args.batches:
            batch += 1
            batch_start_iso = datetime.now().isoformat()
            t0 = time.time()
            waveforms = acquire_batch(inst, args.frames, args.record_length)
            dt = time.time() - t0

            # WFMOUTPRE only reads back correctly once there's real data on
            # the scope, so grab it after the first batch and save it once.
            if "ymult" not in metadata["channels"][str(TRIGGER_CHANNEL)]:
                metadata = get_full_metadata(inst, include_waveform_scaling=True)
                metadata_path.write_text(json.dumps(metadata, indent=2))

            outfile = outdir / f"batch{batch:05d}.root"
            save_batch_root(outfile, waveforms, batch_start_iso, dt)
            print(f"batch {batch}: {args.frames} events in {dt:.1f}s "
                  f"({args.frames/dt:.1f} evt/s) -> {outfile}")
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C).")
    finally:
        inst.write("ACQ:STATE STOP")
        inst.close()


if __name__ == "__main__":
    main()
