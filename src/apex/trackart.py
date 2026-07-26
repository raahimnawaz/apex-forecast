"""Generate the hero artwork from real telemetry.

The track outline on this page is not an illustration of a circuit — it is the actual
GPS trace of the fastest qualifying lap, with the speed channel carried alongside it.
Nothing is drawn by hand and nothing is stylised into inaccuracy: if the line bends, the
car bent there.

The output is a normalised path plus a per-point speed series, ready to be rendered as
SVG. Keeping the geometry and the rendering separate means the art can be restyled
without re-deriving it, and the same payload drives any circuit on the calendar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class TrackArt:
    circuit: str
    driver: str
    lap_time: str
    session: str
    points: list[list[float]]   # [[x, y], ...] normalised into a 0-1000 box
    speed: list[float]          # 0-1 normalised, aligned with points
    speed_kph: list[float]
    fastest_ix: int
    slowest_ix: int
    width: float
    height: float


def _resample(x: np.ndarray, y: np.ndarray, v: np.ndarray, n: int) -> tuple[np.ndarray, ...]:
    """Resample the trace to n points evenly spaced *by distance travelled*.

    Telemetry arrives evenly spaced in time, which means the samples bunch up in slow
    corners and thin out on straights — exactly backwards for drawing, since the corners
    are where the shape lives. Re-spacing by arc length gives an even line weight.
    """
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    if d[-1] <= 0:
        return x, y, v
    t = np.linspace(0, d[-1], n)
    return np.interp(t, d, x), np.interp(t, d, y), np.interp(t, d, v)


def build(session, box: float = 1000.0, n_points: int = 320) -> TrackArt:
    """Derive the artwork payload from a loaded FastF1 session."""
    lap = session.laps.pick_fastest()
    tel = lap.get_telemetry()

    x = tel["X"].to_numpy(dtype=float)
    y = tel["Y"].to_numpy(dtype=float)
    v = tel["Speed"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[ok], y[ok], v[ok]

    # Close the loop so the circuit joins at the start/finish line.
    if np.hypot(x[0] - x[-1], y[0] - y[-1]) > 1e-6:
        x = np.append(x, x[0]); y = np.append(y, y[0]); v = np.append(v, v[0])

    x, y, v = _resample(x, y, v, n_points)

    # Normalise into the box, preserving aspect ratio — a squashed circuit is a wrong one.
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    span = max(x1 - x0, y1 - y0)
    sx = (x - x0) / span * box
    sy = (y - y0) / span * box
    sy = (y1 - y0) / span * box - sy          # SVG y grows downward

    vmin, vmax = float(v.min()), float(v.max())
    vn = (v - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(v)

    return TrackArt(
        circuit=str(session.event["Location"]),
        driver=str(lap["Driver"]),
        lap_time=str(lap["LapTime"]).split("days")[-1].strip()[:-3],
        session=str(session.name),
        points=[[round(float(a), 2), round(float(b), 2)] for a, b in zip(sx, sy)],
        speed=[round(float(s), 4) for s in vn],
        speed_kph=[round(float(s), 1) for s in v],
        fastest_ix=int(np.argmax(v)),
        slowest_ix=int(np.argmin(v)),
        width=round(float((x1 - x0) / span * box), 2),
        height=round(float((y1 - y0) / span * box), 2),
    )


def to_dict(art: TrackArt) -> dict:
    return asdict(art)
