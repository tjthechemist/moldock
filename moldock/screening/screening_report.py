"""
Reporting for virtual screening campaigns: terminal tables, CSV export, and
publication-quality PNG summary dashboards.

PNG dashboard layout (1200 × 900 px)
--------------------------------------
┌─────────────────────────────────────────────────────────────────────┐
│  Header bar: campaign title, receptor name, date/time               │
├──────────────────────────────┬──────────────────────────────────────┤
│  Score histogram             │  Top-N ranked table                  │
│  (distribution of all scores)│  name | score | rot_bonds | ...      │
├──────────────────────────────┴──────────────────────────────────────┤
│  Cumulative hit-rate curve                                          │
├─────────────────────────────────────────────────────────────────────┤
│  Footer stats: total / hits / failures / best                       │
└─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import csv
import math
import os
from datetime import datetime
from typing import List, Optional

import numpy as np

from moldock.screening.virtual_screener import ScreeningCampaign, ScreeningResult

# Pillow is already a dependency
try:
    from PIL import Image, ImageDraw, ImageFont

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Color palette (matches visualizer.py)
# ---------------------------------------------------------------------------
_C = {
    "bg": (255, 255, 255),
    "header": (30, 58, 138),        # Deep indigo
    "header_text": (255, 255, 255),
    "panel": (248, 250, 252),
    "border": (203, 213, 225),
    "text": (30, 41, 59),
    "muted": (100, 116, 139),
    "hit": (22, 163, 74),           # Green
    "miss": (239, 68, 68),          # Red
    "bar_fill": (59, 130, 246),     # Blue
    "bar_best": (22, 163, 74),      # Green for best score
    "curve": (245, 158, 11),        # Amber
    "grid_line": (226, 232, 240),
    "rank1": (253, 230, 138),       # Gold tint for rank 1
}


def _font(size: int = 14, bold: bool = False) -> "ImageFont.FreeTypeFont":
    paths_bold = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    paths_reg = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in (paths_bold if bold else paths_reg):
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _rounded_rect(
    draw: "ImageDraw.ImageDraw",
    box: tuple,
    radius: int,
    fill: tuple,
    outline: Optional[tuple] = None,
    outline_width: int = 1,
):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=fill, outline=outline, width=outline_width)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ScreeningReporter:
    """Generate textual and visual reports from a :class:`ScreeningCampaign`."""

    # ------------------------------------------------------------------
    # Terminal table
    # ------------------------------------------------------------------

    @staticmethod
    def generate_table(
        campaign: ScreeningCampaign,
        top_n: int = 20,
        score_cutoff: Optional[float] = None,
    ) -> str:
        """Return a formatted ASCII leaderboard of the top *top_n* hits."""
        hits = campaign.ranked_hits(top_n=top_n, score_cutoff=score_cutoff)
        total = campaign.total_ligands
        failed = len(campaign.failed)

        lines = []
        lines.append("=" * 90)
        lines.append(f"  VIRTUAL SCREENING RESULTS  —  Receptor: {campaign.receptor_name}")
        lines.append(f"  Time: {campaign.elapsed_seconds:.1f}s  |  Total: {total}  |  "
                     f"Failed: {failed}  |  Hits shown: {len(hits)}")
        lines.append("=" * 90)
        lines.append(
            f"{'Rank':<5} {'Ligand Name':<30} {'Score (kcal/mol)':<18} "
            f"{'RotBonds':<10} {'Vina':<10} {'Metal':<10} {'Elec':<8}"
        )
        lines.append("-" * 90)

        for rank, r in enumerate(hits, start=1):
            bd = r.score_breakdown
            vina = bd.get("vina_score", r.top_score)
            metal = bd.get("metal_score", 0.0)
            elec = bd.get("electrostatic_score", 0.0)
            lines.append(
                f"{rank:<5} {r.ligand_name:<30} {r.top_score:<18.3f} "
                f"{r.num_rotatable_bonds:<10} {vina:<10.3f} {metal:<10.3f} {elec:<8.3f}"
            )

        lines.append("=" * 90)

        if campaign.failed:
            lines.append(f"\n  Failed ligands ({failed}):")
            for r in campaign.failed[:10]:
                lines.append(f"    - {r.ligand_name}: {r.error}")
            if failed > 10:
                lines.append(f"    … and {failed - 10} more.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    @staticmethod
    def export_csv(
        campaign: ScreeningCampaign,
        csv_path: str,
        include_failed: bool = True,
    ) -> None:
        """Write ranked results to a CSV file.

        Columns
        -------
        Rank, LigandName, Score_kcal_mol, VinaScore, MetalScore,
        ElectrostaticScore, NumRotatableBonds, ElapsedSeconds, Error,
        + any SDF data fields found in the first molecule.
        """
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

        # Collect all SDF data keys across all results
        sdf_keys: list = []
        for r in campaign.results:
            for k in r.sdf_data:
                if k not in sdf_keys:
                    sdf_keys.append(k)

        header = [
            "Rank", "LigandName", "Score_kcal_mol", "VinaScore",
            "MetalScore", "ElectrostaticScore", "NumRotatableBonds",
            "ElapsedSeconds", "Status", "Error",
        ] + sdf_keys

        hits = campaign.ranked_hits(successful_only=False)
        hits.sort(key=lambda r: (not r.success, r.top_score))

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            rank = 0
            for r in hits:
                if not include_failed and not r.success:
                    continue
                if r.success:
                    rank += 1

                bd = r.score_breakdown
                row = [
                    rank if r.success else "",
                    r.ligand_name,
                    round(r.top_score, 4) if r.success else "",
                    round(bd.get("vina_score", r.top_score), 4) if r.success else "",
                    round(bd.get("metal_score", 0.0), 4) if r.success else "",
                    round(bd.get("electrostatic_score", 0.0), 4) if r.success else "",
                    r.num_rotatable_bonds,
                    round(r.elapsed_seconds, 2),
                    "OK" if r.success else "FAILED",
                    r.error or "",
                ] + [r.sdf_data.get(k, "") for k in sdf_keys]
                writer.writerow(row)

    # ------------------------------------------------------------------
    # PNG dashboard
    # ------------------------------------------------------------------

    @staticmethod
    def render_dashboard(
        campaign: ScreeningCampaign,
        output_png_path: str,
        top_n: int = 15,
        score_cutoff: Optional[float] = None,
        title: str = "Virtual Screening Dashboard",
    ) -> None:
        """Render a multi-panel PNG summary dashboard.

        Parameters
        ----------
        campaign:
            The completed screening campaign.
        output_png_path:
            Destination ``.png`` path.
        top_n:
            Number of top hits to show in the ranked table panel.
        score_cutoff:
            Optional score threshold line drawn on the histogram.
        title:
            Dashboard title text.
        """
        if not _PIL_AVAILABLE:
            raise RuntimeError("Pillow is required for PNG rendering: pip install pillow")

        W, H = 1300, 980
        img = Image.new("RGB", (W, H), _C["bg"])
        draw = ImageDraw.Draw(img)

        hits = campaign.ranked_hits(top_n=None, score_cutoff=None)
        all_scores = [r.top_score for r in hits]

        # ---- Header -------------------------------------------------------
        draw.rectangle([(0, 0), (W, 64)], fill=_C["header"])
        draw.text((20, 10), title, fill=_C["header_text"], font=_font(22, bold=True))
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        draw.text(
            (W - 260, 10),
            f"Receptor: {campaign.receptor_name}\n{ts}",
            fill=(180, 200, 255),
            font=_font(12),
        )

        # ---- Stats strip --------------------------------------------------
        stats_y = 64
        stats = [
            ("Total Ligands", str(campaign.total_ligands)),
            ("Successful", str(len(hits))),
            ("Failed", str(len(campaign.failed))),
            ("Best Score", f"{hits[0].top_score:.2f} kcal/mol" if hits else "N/A"),
            ("Wall Time", f"{campaign.elapsed_seconds:.1f}s"),
        ]
        sw = W // len(stats)
        for i, (label, val) in enumerate(stats):
            bx0, bx1 = i * sw, (i + 1) * sw - 4
            _rounded_rect(draw, (bx0 + 6, stats_y + 4, bx1 - 6, stats_y + 54),
                           radius=6, fill=_C["panel"], outline=_C["border"])
            draw.text((bx0 + 16, stats_y + 8), label, fill=_C["muted"], font=_font(11))
            draw.text((bx0 + 16, stats_y + 26), val, fill=_C["text"], font=_font(15, bold=True))

        content_y = stats_y + 62

        # ---- Score histogram (left panel) ---------------------------------
        hist_x0, hist_y0 = 20, content_y + 10
        hist_x1, hist_y1 = 580, content_y + 360

        _rounded_rect(draw, (hist_x0, hist_y0, hist_x1, hist_y1),
                       radius=8, fill=_C["panel"], outline=_C["border"])
        draw.text((hist_x0 + 16, hist_y0 + 10), "Score Distribution", fill=_C["text"], font=_font(14, bold=True))

        if all_scores:
            n_bins = min(20, max(5, len(all_scores) // 2))
            lo, hi = min(all_scores), max(all_scores)
            if abs(hi - lo) < 1e-6:
                lo, hi = lo - 1, hi + 1
            bin_w = (hi - lo) / n_bins
            bins = [0] * n_bins
            for s in all_scores:
                idx = min(int((s - lo) / bin_w), n_bins - 1)
                bins[idx] += 1

            max_count = max(bins) or 1
            px0, px1 = hist_x0 + 50, hist_x1 - 20
            py0, py1 = hist_y0 + 40, hist_y1 - 50
            plot_w = px1 - px0
            plot_h = py1 - py0

            # Axes
            draw.line([(px0, py0), (px0, py1), (px1, py1)], fill=_C["text"], width=2)

            bar_w = max(2, int(plot_w / n_bins) - 2)
            for i, count in enumerate(bins):
                bh = int(count / max_count * plot_h)
                bx = px0 + i * (plot_w // n_bins)
                col = _C["bar_best"] if i == 0 else _C["bar_fill"]
                draw.rectangle([(bx, py1 - bh), (bx + bar_w, py1)], fill=col)
                # Count label
                if count > 0:
                    draw.text((bx + 2, py1 - bh - 14), str(count), fill=_C["muted"], font=_font(9))

            # X-axis labels
            for i in range(0, n_bins + 1, max(1, n_bins // 5)):
                val = lo + i * bin_w
                lx = px0 + i * (plot_w // n_bins)
                draw.text((lx - 10, py1 + 4), f"{val:.1f}", fill=_C["muted"], font=_font(9))

            # Y-axis label
            draw.text((hist_x0 + 4, (py0 + py1) // 2 - 30), "Count", fill=_C["muted"], font=_font(10))
            draw.text((px0 + plot_w // 2 - 40, hist_y1 - 20),
                       "Score (kcal/mol)", fill=_C["muted"], font=_font(10))

            # Cutoff line
            if score_cutoff is not None and lo <= score_cutoff <= hi:
                cx = int(px0 + (score_cutoff - lo) / (hi - lo) * plot_w)
                draw.line([(cx, py0), (cx, py1)], fill=_C["miss"], width=2)
                draw.text((cx + 2, py0), f"cutoff={score_cutoff:.1f}", fill=_C["miss"], font=_font(9))

        # ---- Ranked table (right panel) -----------------------------------
        tbl_x0, tbl_y0 = 600, content_y + 10
        tbl_x1, tbl_y1 = W - 20, content_y + 360

        _rounded_rect(draw, (tbl_x0, tbl_y0, tbl_x1, tbl_y1),
                       radius=8, fill=_C["panel"], outline=_C["border"])
        draw.text((tbl_x0 + 16, tbl_y0 + 10),
                   f"Top-{top_n} Ranked Hits", fill=_C["text"], font=_font(14, bold=True))

        col_x = [tbl_x0 + 10, tbl_x0 + 44, tbl_x0 + 200, tbl_x0 + 340, tbl_x0 + 440, tbl_x0 + 530]
        col_labels = ["#", "Name", "Score", "Vina", "Metal", "RotB"]
        row_h = 22
        header_y = tbl_y0 + 36

        draw.rectangle([(tbl_x0 + 6, header_y), (tbl_x1 - 6, header_y + row_h)], fill=_C["header"])
        for cx, cl in zip(col_x, col_labels):
            draw.text((cx, header_y + 4), cl, fill=_C["header_text"], font=_font(11, bold=True))

        top_hits = campaign.ranked_hits(top_n=top_n, score_cutoff=score_cutoff)
        for rank, r in enumerate(top_hits, start=1):
            ry = header_y + rank * row_h
            if ry + row_h > tbl_y1 - 8:
                break
            row_fill = _C["rank1"] if rank == 1 else (_C["bg"] if rank % 2 == 0 else _C["panel"])
            draw.rectangle([(tbl_x0 + 6, ry), (tbl_x1 - 6, ry + row_h - 1)], fill=row_fill)

            bd = r.score_breakdown
            vina = bd.get("vina_score", r.top_score)
            metal = bd.get("metal_score", 0.0)
            vals = [
                str(rank),
                (r.ligand_name[:22] + "…") if len(r.ligand_name) > 23 else r.ligand_name,
                f"{r.top_score:.3f}",
                f"{vina:.3f}",
                f"{metal:.3f}",
                str(r.num_rotatable_bonds),
            ]
            score_color = _C["hit"] if r.top_score < -5 else (_C["miss"] if r.top_score > -2 else _C["text"])
            for ci, (cx, v) in enumerate(zip(col_x, vals)):
                color = score_color if ci == 2 else _C["text"]
                draw.text((cx, ry + 4), v, fill=color, font=_font(10))

        # ---- Cumulative hit-rate curve ------------------------------------
        curve_x0, curve_y0 = 20, content_y + 380
        curve_x1, curve_y1 = W - 20, content_y + 620

        _rounded_rect(draw, (curve_x0, curve_y0, curve_x1, curve_y1),
                       radius=8, fill=_C["panel"], outline=_C["border"])
        draw.text((curve_x0 + 16, curve_y0 + 10),
                   "Cumulative Score Profile", fill=_C["text"], font=_font(14, bold=True))

        if len(all_scores) >= 2:
            sorted_scores = sorted(all_scores)
            px0, px1 = curve_x0 + 60, curve_x1 - 30
            py0, py1 = curve_y0 + 40, curve_y1 - 40
            pw, ph = px1 - px0, py1 - py0
            n = len(sorted_scores)

            # Grid lines
            for gi in range(0, 6):
                gy = py0 + gi * ph // 5
                draw.line([(px0, gy), (px1, gy)], fill=_C["grid_line"], width=1)

            # Axes
            draw.line([(px0, py0), (px0, py1), (px1, py1)], fill=_C["text"], width=2)

            s_min, s_max = sorted_scores[0], sorted_scores[-1]
            if abs(s_max - s_min) < 1e-6:
                s_min -= 1
                s_max += 1

            pts = []
            for i, s in enumerate(sorted_scores):
                cx_ = int(px0 + i / (n - 1) * pw) if n > 1 else px0
                cy_ = int(py0 + (1.0 - (s - s_min) / (s_max - s_min)) * ph)
                pts.append((cx_, cy_))

            for i in range(len(pts) - 1):
                draw.line([pts[i], pts[i + 1]], fill=_C["curve"], width=3)

            # X-axis labels (ligand rank)
            for gi in range(0, 6):
                gx = px0 + gi * pw // 5
                label_i = int(gi * (n - 1) / 5)
                draw.text((gx - 12, py1 + 4), str(label_i + 1), fill=_C["muted"], font=_font(9))

            # Y-axis labels (score)
            for gi in range(0, 6):
                gy = py0 + gi * ph // 5
                score_val = s_max - gi * (s_max - s_min) / 5
                draw.text((curve_x0 + 4, gy - 6), f"{score_val:.1f}", fill=_C["muted"], font=_font(9))

            draw.text((px0 + pw // 2 - 40, py1 + 18), "Ligand Rank (sorted by score)", fill=_C["muted"], font=_font(10))
            draw.text((curve_x0 + 4, py0 + ph // 2 - 20), "Score\n(kcal/\nmol)", fill=_C["muted"], font=_font(9))

        # ---- Footer -------------------------------------------------------
        footer_y = content_y + 640
        draw.rectangle([(0, footer_y), (W, H)], fill=_C["panel"])
        draw.line([(0, footer_y), (W, footer_y)], fill=_C["border"], width=1)

        footer_items = [
            f"Receptor: {campaign.receptor_name}",
            f"Exhaustiveness: {campaign.config_summary.get('exhaustiveness', '?')}",
            f"Modes per ligand: {campaign.config_summary.get('num_modes', '?')}",
            f"Score cutoff: {campaign.config_summary.get('score_cutoff', 'None')}",
            f"MolDock Virtual Screener",
        ]
        fx = 20
        for item in footer_items:
            draw.text((fx, footer_y + 12), item, fill=_C["muted"], font=_font(11))
            fw = draw.textlength(item, font=_font(11))
            fx += int(fw) + 40

        os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
        img.save(output_png_path, format="PNG", optimize=True)

    # ------------------------------------------------------------------
    # SDF export of top hits with docking scores embedded as data fields
    # ------------------------------------------------------------------

    @staticmethod
    def export_hits_sdf(
        campaign: ScreeningCampaign,
        sdf_path: str,
        top_n: Optional[int] = None,
        score_cutoff: Optional[float] = None,
    ) -> None:
        """Write top docked poses to a multi-molecule SDF file.

        Each record includes the docked coordinates and scores as SDF
        data fields (``> <MOLDOCK_SCORE>``, etc.), making the file
        compatible with any cheminformatics toolkit.
        """
        from moldock.core.sdf_io import SDFIO

        hits = campaign.ranked_hits(top_n=top_n, score_cutoff=score_cutoff)
        if not hits:
            print("[WARN] No hits to export.")
            return

        os.makedirs(os.path.dirname(os.path.abspath(sdf_path)), exist_ok=True)

        with open(sdf_path, "w", encoding="utf-8") as fh:
            for rank, sr in enumerate(hits, start=1):
                if sr.top_pose is None or sr.docking_result is None:
                    continue

                # Write V2000 coordinate block
                mol = sr.top_pose.apply_to_molecule(sr.docking_result.query)
                atoms = mol.atoms
                bonds = mol.metadata.get("bonds", [])
                bond_orders = mol.metadata.get("bond_orders", {})

                fh.write(f"{mol.name}\n  moldock virtual screening  rank={rank}\n\n")
                n_a, n_b = len(atoms), len(bonds)
                fh.write(f"{n_a:3d}{n_b:3d}  0  0  0  0  0  0  0  0999 V2000\n")

                _CHARGE_WRITE = {3.0: 1, 2.0: 2, 1.0: 3, -1.0: 5, -2.0: 6, -3.0: 7}
                for atom in atoms:
                    x, y, z = atom.coord
                    chg_raw = _CHARGE_WRITE.get(round(atom.charge), 0)
                    fh.write(
                        f"{x:10.4f}{y:10.4f}{z:10.4f} {atom.element:<3}  0 {chg_raw:2d}"
                        f"  0  0  0  0  0  0  0  0  0  0\n"
                    )
                for a, b in bonds:
                    order = bond_orders.get((a, b), bond_orders.get((b, a), 1))
                    fh.write(f"{a + 1:3d}{b + 1:3d}{order:3d}  0  0  0  0\n")
                fh.write("M  END\n")

                # SDF data fields
                bd = sr.score_breakdown
                fields = {
                    "MOLDOCK_RANK": str(rank),
                    "MOLDOCK_SCORE": f"{sr.top_score:.4f}",
                    "MOLDOCK_VINA_SCORE": f"{bd.get('vina_score', sr.top_score):.4f}",
                    "MOLDOCK_METAL_SCORE": f"{bd.get('metal_score', 0.0):.4f}",
                    "MOLDOCK_ELEC_SCORE": f"{bd.get('electrostatic_score', 0.0):.4f}",
                    "MOLDOCK_ROT_BONDS": str(sr.num_rotatable_bonds),
                    **sr.sdf_data,
                }
                for tag, val in fields.items():
                    fh.write(f"> <{tag}>\n{val}\n\n")

                fh.write("$$$$\n")

