"""
Visual analysis and PNG rendering for molecular docking results:
2D Interaction Networks, 3D Binding Site Projections, and Multi-Panel Summary Dashboards.
"""

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moldock.core.atom import Atom
from moldock.core.molecule import Molecule
from moldock.metals.detector import MetalCenter
from moldock.analysis.interactions import InteractionContact, InteractionProfiler
from moldock.analysis.bond_orders import (
    AROMATIC_ORDER,
    DOUBLE,
    SINGLE,
    TRIPLE,
    BondKey,
    bond_key,
    build_adjacency,
    find_rings,
    infer_bonds_from_geometry,
    perceive_bond_orders,
)
from moldock.analysis.ligand_layout import layout_ligand_depiction


# Professional CPK and Interaction Color Palette
COLORS = {
    "bg": (255, 255, 255),
    "panel_bg": (248, 250, 252),
    "border": (226, 232, 240),
    "text_dark": (30, 41, 59),
    "text_muted": (100, 116, 139),
    # Elements (CPK scheme)
    "C":  (60,  60,  60),
    "A":  (60,  60,  60),   # aromatic carbon — same shade
    "N":  (49,  130, 206),
    "O":  (229, 62,  62),
    "S":  (214, 158, 46),
    "H":  (160, 174, 192),
    "P":  (237, 137, 54),
    "CL": (56,  161, 105),
    "F":  (72,  187, 120),
    "BR": (155, 44,  44),
    "I":  (128, 0,   128),
    # Metals
    "ZN": (128, 90,  213),
    "MG": (107, 70,  193),
    "CA": (142, 68,  173),
    "FE": (197, 48,  48),
    "CU": (49,  151, 149),
    "MN": (159, 122, 234),
    # Interactions
    "metal_coordination": (159, 122, 234),
    "hbond":              (56,  161, 105),
    "salt_bridge":        (221, 107, 32),
    "hydrophobic":        (113, 128, 150),
}

# CPK atom colours used for the ligand 2D depiction
_ELEM_COLOR = {
    "C":  (50,  50,  50),
    "A":  (50,  50,  50),
    "N":  (49,  130, 206),
    "O":  (220, 50,  50),
    "S":  (200, 150, 30),
    "P":  (237, 137, 54),
    "F":  (50,  180, 90),
    "CL": (50,  160, 80),
    "BR": (140, 40,  40),
    "I":  (100, 0,   100),
    "H":  (200, 200, 200),
}

# Ligand depiction styling
BOND_COLOR = (105, 112, 128)
HIGHLIGHT_COLOR = (255, 215, 50)
HIGHLIGHT_PAD = 4
# Distance badges are parked on the interaction line just outside the ligand
# glyph area, so they never cover the drawn structure.
BADGE_CLEARANCE = 45.0          # pixels beyond the ligand ellipse
BADGE_HALF_WIDTH = 19
_BOND_ORDER_OFFSET = 3          # pixel offset for double/triple bond lines
BOND_LINE_WIDTH = 2
AROMATIC_INNER_FRACTION = 0.64  # inner aromatic line spans this fraction of the bond
AROMATIC_MAX_OFFSET = 5

# Atom glyph sizing, expressed as a fraction of the on-screen bond length.
# The layout guarantees that no two atoms sit closer than
# ``ligand_layout.MIN_SEPARATION_FACTOR`` (0.62) of a bond length, so keeping
# the glyph radius at 0.28 of a bond length means a glyph diameter of 0.56 of a
# bond — always smaller than the closest allowed spacing.  That keeps every
# atom of the ligand visible, with no atom hiding another, at any molecule size.
ATOM_RADIUS_FACTOR = 0.28
# Degenerate floor so the ellipse never collapses to zero pixels
MIN_ATOM_RADIUS = 1.0
MAX_ATOM_RADIUS = 16.0
MIN_ATOM_FONT = 8
MAX_ATOM_FONT = 13
LABEL_FIT_RATIO = 1.75
# Below this on-screen bond length a depiction is too dense to read anyway
LEGIBLE_BOND_PIXELS = 8.0


def _get_font(size: int = 14) -> Any:
    """Attempt to load a clean TrueType font or fall back to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in font_paths:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    fill: Tuple[int, int, int],
    width: int = 2,
    dash_length: int = 6,
    space_length: int = 4,
):
    """Draw a clean dashed line between two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1e-4:
        return

    ux = dx / dist
    uy = dy / dist
    curr = 0.0

    while curr < dist:
        seg_end = min(curr + dash_length, dist)
        sx1 = x1 + ux * curr
        sy1 = y1 + uy * curr
        sx2 = x1 + ux * seg_end
        sy2 = y1 + uy * seg_end
        draw.line([(sx1, sy1), (sx2, sy2)], fill=fill, width=width)
        curr += dash_length + space_length


def _draw_rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    fill: Tuple[int, int, int],
    outline: Optional[Tuple[int, int, int]] = None,
    radius: int = 8,
    width: int = 1,
):
    """Draw a rounded rectangle."""
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


# ---------------------------------------------------------------------------
# 2D ligand depiction helpers
# ---------------------------------------------------------------------------

def atom_radius_for_bond(bond_pixels: float) -> float:
    """Glyph radius that keeps every atom separate at this bond length."""
    return float(min(MAX_ATOM_RADIUS,
                     max(MIN_ATOM_RADIUS, bond_pixels * ATOM_RADIUS_FACTOR)))


def _badge_position(
    p_from: Tuple[float, float],
    p_to: Tuple[float, float],
    center: Tuple[float, float],
    min_radius: float,
    min_fraction: float = 0.28,
    max_fraction: float = 0.86,
) -> Tuple[float, float]:
    """Point on the segment p_from→p_to that first leaves a circle.

    Distance badges are placed with this so they sit in the gap between the
    ligand depiction and the receptor nodes instead of covering the structure.
    The exact exit point is found by solving ``|p_from + t·u − center| = r``;
    when the segment never reaches the radius the midpoint is used instead.
    """
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return p_from

    ux, uy = dx / length, dy / length
    cx = p_from[0] - center[0]
    cy = p_from[1] - center[1]

    b = cx * ux + cy * uy
    c = cx * cx + cy * cy - min_radius * min_radius
    discriminant = b * b - c

    if discriminant >= 0.0:
        root = math.sqrt(discriminant)
        t = -b + root
        if t < 0.0:
            t = -b - root
        fraction = t / length
    else:
        fraction = 0.55

    fraction = min(max(fraction, min_fraction), max_fraction)
    return (p_from[0] + dx * fraction, p_from[1] + dy * fraction)


def _unit_vector(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> Tuple[float, float, float]:
    """Return the unit direction from p1 to p2 together with the distance."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0, 0.0, 0.0
    return dx / dist, dy / dist, dist


def _ring_centroids(
    positions: Sequence[Tuple[float, float]],
    rings: Sequence[Sequence[int]],
    aromatic_bonds: Set[BondKey],
) -> Dict[BondKey, Tuple[float, float]]:
    """Map every aromatic bond to the 2D centroid of the ring it belongs to."""
    centroids: Dict[BondKey, Tuple[float, float]] = {}
    for ring in rings:
        if len(ring) < 3:
            continue
        members = [bond_key(ring[k], ring[(k + 1) % len(ring)]) for k in range(len(ring))]
        if not all(member in aromatic_bonds for member in members):
            continue
        cx = sum(positions[a][0] for a in ring) / len(ring)
        cy = sum(positions[a][1] for a in ring) / len(ring)
        for member in members:
            centroids[member] = (cx, cy)
    return centroids


def _draw_aromatic_inner(
    draw: ImageDraw.ImageDraw,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    inner_point: Optional[Tuple[float, float]],
    color: Tuple[int, int, int],
):
    """Draw the shortened second line that marks an aromatic bond."""
    ux, uy, dist = _unit_vector(p1, p2)
    if dist < 1e-6:
        return

    nx, ny = -uy, ux
    mid_x = (p1[0] + p2[0]) / 2.0
    mid_y = (p1[1] + p2[1]) / 2.0

    if inner_point is not None:
        # Flip the offset towards the ring interior
        side = (inner_point[0] - mid_x) * nx + (inner_point[1] - mid_y) * ny
        if side < 0:
            nx, ny = -nx, -ny

    offset = min(_BOND_ORDER_OFFSET + 1, AROMATIC_MAX_OFFSET, dist * 0.20)
    half = dist * AROMATIC_INNER_FRACTION / 2.0
    start = (mid_x + nx * offset - ux * half, mid_y + ny * offset - uy * half)
    end = (mid_x + nx * offset + ux * half, mid_y + ny * offset + uy * half)
    draw.line([start, end], fill=color, width=BOND_LINE_WIDTH)


def _draw_bond(
    draw: ImageDraw.ImageDraw,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    order: int,
    inner_point: Optional[Tuple[float, float]] = None,
):
    """Render a single bond with the multiplicity implied by *order*."""
    ux, uy, dist = _unit_vector(p1, p2)
    if dist < 1e-6:
        return

    nx, ny = -uy, ux

    if order == DOUBLE:
        o = min(_BOND_ORDER_OFFSET, dist * 0.18)
        draw.line([(p1[0] + nx * o, p1[1] + ny * o),
                   (p2[0] + nx * o, p2[1] + ny * o)], fill=BOND_COLOR, width=BOND_LINE_WIDTH)
        draw.line([(p1[0] - nx * o, p1[1] - ny * o),
                   (p2[0] - nx * o, p2[1] - ny * o)], fill=BOND_COLOR, width=BOND_LINE_WIDTH)
    elif order == TRIPLE:
        draw.line([p1, p2], fill=BOND_COLOR, width=BOND_LINE_WIDTH)
        o = min(_BOND_ORDER_OFFSET + 1, dist * 0.22)
        draw.line([(p1[0] + nx * o, p1[1] + ny * o),
                   (p2[0] + nx * o, p2[1] + ny * o)], fill=BOND_COLOR, width=1)
        draw.line([(p1[0] - nx * o, p1[1] - ny * o),
                   (p2[0] - nx * o, p2[1] - ny * o)], fill=BOND_COLOR, width=1)
    elif order == AROMATIC_ORDER:
        draw.line([p1, p2], fill=BOND_COLOR, width=BOND_LINE_WIDTH)
        _draw_aromatic_inner(draw, p1, p2, inner_point, BOND_COLOR)
    else:
        draw.line([p1, p2], fill=BOND_COLOR, width=BOND_LINE_WIDTH)


def _atom_label(element: str, atom_count: int) -> str:
    """Text drawn on an atom glyph (empty for skeletal carbons)."""
    elem = element.upper()
    if elem in ("C", "A"):
        return "C" if atom_count == 1 else ""
    return elem


def _draw_atom_label(
    draw: ImageDraw.ImageDraw,
    pos: Tuple[float, float],
    label: str,
    radius: float,
    color: Tuple[int, int, int],
    font: Any,
):
    """Draw an element label inside the glyph, or beside it when too small."""
    try:
        bbox = font.getbbox(label)
    except AttributeError:      # very old Pillow fallback font
        bbox = (0, 0, len(label) * 6, 12)

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    if text_w <= radius * LABEL_FIT_RATIO and text_h <= radius * LABEL_FIT_RATIO:
        text_color = (255, 255, 255) if sum(color) < 400 else (25, 25, 25)
        draw.text((pos[0] - text_w / 2.0 - bbox[0], pos[1] - text_h / 2.0 - bbox[1]),
                  label, fill=text_color, font=font)
    else:
        # Glyph too small to hold the text — place it just outside instead
        draw.text((pos[0] + radius + 2, pos[1] - text_h / 2.0 - bbox[1]),
                  label, fill=(40, 40, 40), font=font)


def _draw_ligand_depiction(
    draw: ImageDraw.ImageDraw,
    atoms: Sequence[Atom],
    positions: Sequence[Tuple[float, float]],
    bonds: Sequence[Tuple[int, int]],
    bond_orders: Dict[BondKey, int],
    interacting_indices: Set[int],
    bond_pixels: float,
    highlight_color: Tuple[int, int, int] = HIGHLIGHT_COLOR,
):
    """Draw **every** ligand atom and bond as a flat 2D chemical structure.

    Atoms taking part in a receptor contact are ringed with *highlight_color*
    so the interaction sites stand out, but no atom is ever omitted — the
    whole molecule is always shown.

    Parameters
    ----------
    positions:
        One 2D pixel position per atom, in the same order as *atoms*.
    bond_pixels:
        On-screen length of a typical bond; drives the glyph radius so no two
        atoms can overlap.
    """
    atom_count = len(atoms)
    if atom_count == 0 or len(positions) != atom_count:
        return

    clean_bonds = [bond_key(i, j) for i, j in bonds
                   if 0 <= i < atom_count and 0 <= j < atom_count and i != j]

    adj = build_adjacency(atom_count, clean_bonds)
    rings = find_rings(adj, clean_bonds)
    aromatic_bonds = {key for key, order in bond_orders.items() if order == AROMATIC_ORDER}
    centroids = _ring_centroids(positions, rings, aromatic_bonds)

    # ---- Bonds first, so atoms sit on top of them --------------------------
    for key in clean_bonds:
        i, j = key
        _draw_bond(draw, positions[i], positions[j],
                   bond_orders.get(key, SINGLE), centroids.get(key))

    # ---- Every atom, with its element label -------------------------------
    radius = atom_radius_for_bond(bond_pixels)
    font_size = int(min(MAX_ATOM_FONT, max(MIN_ATOM_FONT, round(radius * 0.9))))
    font = _get_font(font_size)

    for idx, atom in enumerate(atoms):
        px, py = positions[idx]
        color = _ELEM_COLOR.get(atom.element.upper(), (110, 110, 110))
        is_interacting = idx in interacting_indices

        if is_interacting:
            outer = radius + HIGHLIGHT_PAD
            draw.ellipse((px - outer, py - outer, px + outer, py + outer),
                         fill=highlight_color)

        draw.ellipse((px - radius, py - radius, px + radius, py + radius),
                     fill=color, outline=(255, 255, 255),
                     width=2 if is_interacting else 1)

        label = _atom_label(atom.element, atom_count)
        if label:
            _draw_atom_label(draw, (px, py), label, radius, color, font)


def _draw_bond_sample(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    order: int,
    length: float = 30.0,
):
    """Small bond glyph used by the bond-order legend."""
    _draw_bond(draw, (x, y), (x + length, y), order, (x + length / 2.0, y - 12.0))


class InteractionVisualizer:
    """
    Renders visual interaction diagrams, 3D projections, and contact networks to PNG.
    """

    @classmethod
    def render_interaction_diagram(
        cls,
        receptor: Molecule,
        query: Molecule,
        contacts: List[InteractionContact],
        output_png_path: str,
        title: str = "Molecular Docking Interaction Map",
        score_info: Optional[Dict[str, Any]] = None,
        width: int = 1400,
        height: int = 900,
    ) -> str:
        """
        Generate a 2D interaction diagram showing the **complete ligand structure**
        surrounded by interacting protein residues.

        Layout
        ------
        • Centre panel  : Full ligand drawn as a flat 2D chemical graph — every
                          atom and every bond, with aromatic rings and double
                          bonds resolved and CPK colours.  Atoms that contact the
                          receptor are ringed in gold.
        • Outer ring    : Receptor residue / metal-cofactor nodes arranged radially.
        • Connecting arcs: Colour-coded dashed lines (H-bond, hydrophobic, …) from
                          each interacting ligand atom to its receptor partner, with
                          a distance label at the midpoint.
        • Right panel   : Legend + numbered contact table.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
        img = Image.new("RGB", (width, height), COLORS["bg"])
        draw = ImageDraw.Draw(img)

        font_title = _get_font(22)
        font_sub   = _get_font(13)
        font_node  = _get_font(12)
        font_dist  = _get_font(11)
        font_small = _get_font(10)

        # ── Header ────────────────────────────────────────────────────────────
        _draw_rounded_rect(draw, (20, 16, width - 20, 75),
                           fill=COLORS["panel_bg"], outline=COLORS["border"], radius=10)
        draw.text((40, 26), title, fill=COLORS["text_dark"], font=font_title)

        subtitle_parts = [
            f"Receptor: {receptor.name}",
            f"Ligand: {query.name}  ({len(query.atoms)} atoms)",
            f"Contacts: {len(contacts)}",
        ]
        if score_info and "total_score" in score_info:
            subtitle_parts.append(f"Affinity: {score_info['total_score']:.2f} kcal/mol")
        if score_info and "metal_score" in score_info and abs(score_info["metal_score"]) > 1e-3:
            subtitle_parts.append(f"Metal: {score_info['metal_score']:.2f} kcal/mol")
        draw.text((40, 52), "  |  ".join(subtitle_parts), fill=COLORS["text_muted"], font=font_sub)

        # ── Layout constants ──────────────────────────────────────────────────
        right_panel_x = 960
        graph_width   = right_panel_x - 20           # ~940 px for the graph area
        graph_cx      = graph_width // 2 + 10        # ~480
        graph_cy      = (height + 75) // 2 + 10      # vertical centre of graph area

        # Orbit radius: residue nodes placed here
        orbit_r = min(graph_width, height - 90) * 0.41

        # Ligand drawing area: a square inset inside the orbit
        lig_half = orbit_r * 0.56   # half-size of ligand canvas region

        # ── Molecular graph of the ligand ─────────────────────────────────────
        atoms = query.atoms
        n_atoms = len(atoms)
        atom_idx_map: Dict[int, int] = {a.index: i for i, a in enumerate(atoms)}

        # Prefer an explicit connection table (SDF/MOL); otherwise infer the
        # graph from interatomic distances (PDB / PDBQT input).
        raw_bonds: List[Tuple[int, int]] = list(query.metadata.get("bonds") or [])
        if not raw_bonds:
            raw_bonds = infer_bonds_from_geometry(atoms)

        # Bond orders: trust an explicit table, otherwise perceive them from the
        # 3D geometry so rings and carbonyl/sulfonyl groups are drawn correctly.
        bond_orders = perceive_bond_orders(
            atoms, raw_bonds, existing=query.metadata.get("bond_orders") or {}
        )

        # Collect 0-based interacting atom indices
        interacting_0idx: Set[int] = set()
        for c in contacts:
            if c.query_atom_idx in atom_idx_map:
                interacting_0idx.add(atom_idx_map[c.query_atom_idx])

        # ── Lay out the complete ligand as a flat 2D depiction ────────────────
        positions, bond_pixels = layout_ligand_depiction(
            atoms=atoms,
            bonds=raw_bonds,
            box_width=float(lig_half * 2),
            box_height=float(lig_half * 2),
            center_x=float(graph_cx),
            center_y=float(graph_cy),
        )

        # ── Draw ligand boundary ellipse (soft grey) ──────────────────────────
        draw.ellipse(
            (graph_cx - lig_half - 26, graph_cy - lig_half - 26,
             graph_cx + lig_half + 26, graph_cy + lig_half + 26),
            outline=(222, 227, 236),
            width=2,
        )
        caption = (f"COMPLETE LIGAND STRUCTURE   ·   {n_atoms} atoms   ·   "
                   f"{len(raw_bonds)} bonds")
        draw.text((graph_cx - 150, graph_cy + lig_half + 32), caption,
                  fill=COLORS["text_muted"], font=font_sub)

        # ── Build receptor partner metadata ───────────────────────────────────
        rec_partners: Dict[str, Dict[str, Any]] = {}
        for c in contacts:
            key = f"{c.receptor_res}_{c.receptor_res_seq}"
            if key not in rec_partners:
                r_atom = next((a for a in receptor.atoms if a.index == c.receptor_atom_idx), None)
                is_metal = r_atom.is_metal if r_atom else (c.interaction_type == "metal_coordination")
                rec_partners[key] = {
                    "label": f"{c.receptor_res}{c.receptor_res_seq}",
                    "is_metal": is_metal,
                    "element": r_atom.element.upper() if r_atom else "M",
                    "contacts": [],
                }
            rec_partners[key]["contacts"].append(c)

        # ── Place receptor nodes on outer orbit ───────────────────────────────
        r_keys = list(rec_partners.keys())
        num_r  = max(len(r_keys), 1)
        r_positions: Dict[str, Tuple[float, float]] = {}
        for i, key in enumerate(r_keys):
            angle = 2.0 * math.pi * i / num_r - math.pi / 2.0
            rx = graph_cx + orbit_r * math.cos(angle)
            ry = graph_cy + orbit_r * math.sin(angle)
            r_positions[key] = (rx, ry)

        # ── Draw interaction lines from ligand atom → receptor node ───────────
        distance_badges: List[Tuple[Tuple[float, float], str, Tuple[int, int, int]]] = []
        for c in contacts:
            q0 = atom_idx_map.get(c.query_atom_idx)
            rec_key = f"{c.receptor_res}_{c.receptor_res_seq}"
            if q0 is None or rec_key not in r_positions or q0 >= len(positions):
                continue

            q_pos = positions[q0]
            r_pos = r_positions[rec_key]

            itype  = c.interaction_type
            color  = COLORS.get(itype, COLORS["text_muted"])

            if itype == "metal_coordination":
                draw.line([r_pos, q_pos], fill=color, width=4)
            else:
                _draw_dashed_line(draw, r_pos, q_pos, fill=color, width=2)

            # Distance badge is parked just outside the ligand glyph area so it
            # never covers the structure drawn beneath it.
            badge_xy = _badge_position(
                q_pos, r_pos,
                center=(float(graph_cx), float(graph_cy)),
                min_radius=lig_half + BADGE_CLEARANCE,
            )
            distance_badges.append((badge_xy, f"{c.distance:.1f}Å", color))

        # ── Draw the complete ligand structure (every atom and bond) ──────────
        _draw_ligand_depiction(
            draw,
            atoms=atoms,
            positions=positions,
            bonds=raw_bonds,
            bond_orders=bond_orders,
            interacting_indices=interacting_0idx,
            bond_pixels=bond_pixels,
        )

        # ── Distance badges, above the ligand so they stay readable ───────────
        for (mx, my), dist_text, color in distance_badges:
            _draw_rounded_rect(
                draw,
                (int(mx - BADGE_HALF_WIDTH), int(my - 9),
                 int(mx + BADGE_HALF_WIDTH), int(my + 9)),
                fill=(255, 255, 255), outline=color, radius=4,
            )
            draw.text((mx - BADGE_HALF_WIDTH + 4, my - 6), dist_text,
                      fill=color, font=font_dist)

        # ── Draw receptor residue / metal nodes ───────────────────────────────
        for key, pos in r_positions.items():
            data = rec_partners[key]

            if data["is_metal"]:
                draw.ellipse(
                    (pos[0] - 28, pos[1] - 28, pos[0] + 28, pos[1] + 28),
                    fill=(243, 232, 255), outline=COLORS["ZN"], width=3,
                )
                draw.ellipse(
                    (pos[0] - 22, pos[1] - 22, pos[0] + 22, pos[1] + 22),
                    fill=COLORS["ZN"], outline=(255, 255, 255), width=2,
                )
                draw.text((pos[0] - 14, pos[1] - 7),
                           data["element"][:2], fill=(255, 255, 255), font=font_node)
                draw.text((pos[0] - 22, pos[1] + 30),
                           f"{data['label']} (Cofactor)", fill=COLORS["ZN"], font=font_sub)
            else:
                # Colour-code the residue badge by dominant interaction type
                dom_type = max(
                    set(c.interaction_type for c in data["contacts"]),
                    key=lambda t: sum(1 for c in data["contacts"] if c.interaction_type == t)
                )
                badge_col = COLORS.get(dom_type, COLORS["border"])
                _draw_rounded_rect(
                    draw,
                    (int(pos[0] - 40), int(pos[1] - 20), int(pos[0] + 40), int(pos[1] + 20)),
                    fill=COLORS["panel_bg"], outline=badge_col, radius=8, width=2,
                )
                draw.text((pos[0] - 30, pos[1] - 8),
                           data["label"], fill=COLORS["text_dark"], font=font_node)

        # ── Right-hand contact panel ──────────────────────────────────────────
        panel_x = right_panel_x
        panel_w = width - panel_x - 20
        _draw_rounded_rect(draw, (panel_x, 90, panel_x + panel_w, height - 20),
                           fill=COLORS["panel_bg"], outline=COLORS["border"], radius=12)

        draw.text((panel_x + 16, 105), "INTERACTION CONTACTS",
                   fill=COLORS["text_dark"], font=font_title)
        draw.line([(panel_x + 16, 138), (panel_x + panel_w - 16, 138)],
                   fill=COLORS["border"], width=1)

        # ---- Interaction legend ----------------------------------------------
        leg_y = 150
        legends = [
            ("Metal Coordination", COLORS["metal_coordination"], True),
            ("Hydrogen Bond",      COLORS["hbond"],              False),
            ("Salt Bridge",        COLORS["salt_bridge"],         False),
            ("Hydrophobic",        COLORS["hydrophobic"],         False),
        ]
        for leg_text, col, is_solid in legends:
            if is_solid:
                draw.line([(panel_x + 16, leg_y + 8), (panel_x + 54, leg_y + 8)], fill=col, width=4)
            else:
                _draw_dashed_line(draw, (panel_x + 16, leg_y + 8), (panel_x + 54, leg_y + 8),
                                   fill=col, width=2)
            draw.text((panel_x + 62, leg_y), leg_text, fill=COLORS["text_dark"], font=font_sub)
            leg_y += 24

        # Interacting-atom marker
        draw.ellipse((panel_x + 16, leg_y + 2, panel_x + 30, leg_y + 16),
                      fill=HIGHLIGHT_COLOR, outline=(200, 160, 20), width=1)
        draw.text((panel_x + 40, leg_y), "Interacting ligand atom",
                  fill=COLORS["text_dark"], font=font_sub)
        leg_y += 22

        # Complete-structure note
        draw.text((panel_x + 16, leg_y),
                  f"Whole ligand drawn: {n_atoms} atoms, {len(raw_bonds)} bonds",
                  fill=COLORS["text_muted"], font=font_sub)
        leg_y += 26

        # ---- Bond-order legend ------------------------------------------------
        for bond_order, bond_text in ((SINGLE, "Single bond"),
                                      (DOUBLE, "Double bond"),
                                      (AROMATIC_ORDER, "Aromatic ring")):
            _draw_bond_sample(draw, panel_x + 16, leg_y + 8, bond_order)
            draw.text((panel_x + 62, leg_y), bond_text,
                      fill=COLORS["text_dark"], font=font_sub)
            leg_y += 22

        draw.line([(panel_x + 16, leg_y + 8), (panel_x + panel_w - 16, leg_y + 8)],
                   fill=COLORS["border"], width=1)
        row_y = leg_y + 22

        # ---- Contact table ----------------------------------------------------
        col_xs = [panel_x + 16, panel_x + 86, panel_x + 190, panel_x + 290, panel_x + 355]
        col_heads = ["#", "Type", "Receptor", "Ligand", "Å"]
        for cx, ch in zip(col_xs, col_heads):
            draw.text((cx, row_y), ch, fill=COLORS["text_muted"], font=font_small)
        row_y += 18

        max_rows = max(1, (height - row_y - 30) // 21)
        for ci, c in enumerate(contacts[:max_rows], start=1):
            row_bg = (240, 245, 255) if ci % 2 == 0 else COLORS["bg"]
            _draw_rounded_rect(draw,
                                (panel_x + 10, row_y - 2, panel_x + panel_w - 10, row_y + 17),
                                fill=row_bg, radius=3)
            type_short = c.interaction_type.replace("_coordination", "").replace("_", " ").upper()
            type_col   = COLORS.get(c.interaction_type, COLORS["text_dark"])
            rec_str    = f"{c.receptor_res}{c.receptor_res_seq}:{c.receptor_atom_name}"
            lig_str    = f"{c.query_atom_name}({c.query_element})"
            vals = [str(ci), type_short[:8], rec_str[:13], lig_str[:12], f"{c.distance:.2f}"]
            colors_row = [COLORS["text_dark"], type_col] + [COLORS["text_dark"]] * 3
            for cx, v, vc in zip(col_xs, vals, colors_row):
                draw.text((cx, row_y), v, fill=vc, font=font_small)
            row_y += 21

        if len(contacts) > max_rows:
            draw.text((panel_x + 16, row_y + 4),
                       f"… {len(contacts) - max_rows} more contacts",
                       fill=COLORS["text_muted"], font=font_small)

        # Save PNG
        img.save(output_png_path, "PNG", dpi=(150, 150))
        return output_png_path

    @classmethod
    def render_3d_pocket_projection(
        cls,
        receptor: Molecule,
        query: Molecule,
        contacts: List[InteractionContact],
        metal_centers: List[MetalCenter],
        output_png_path: str,
        title: str = "3D Binding Pocket & Metal Coordination",
        width: int = 1200,
        height: int = 900,
    ) -> str:
        """
        Orthographic 3D projection of the active site pocket:
        Renders atoms with CPK colors, bonds as solid sticks,
        metal cofactors as prominent spheres, and coordination bonds as dashed rays.

        Every ligand atom is labelled with its element so the complete docked
        structure can be read even where the projection overlaps atoms.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
        img = Image.new("RGB", (width, height), (15, 23, 42))  # Elegant Dark Slate background
        draw = ImageDraw.Draw(img)

        font_title = _get_font(22)
        font_sub = _get_font(13)
        font_atom = _get_font(11)

        draw.text((40, 30), title, fill=(248, 250, 252), font=font_title)
        draw.text((40, 60), f"Receptor: {receptor.name}  |  Query: {query.name}  |  Metals: {len(metal_centers)}", fill=(148, 163, 184), font=font_sub)

        # Extract atoms in binding pocket (near query or metals)
        q_coords = query.coords
        if len(q_coords) == 0:
            img.save(output_png_path, "PNG")
            return output_png_path

        q_center = np.mean(q_coords, axis=0)

        pocket_rec_atoms = []
        for a in receptor.atoms:
            d = float(np.linalg.norm(a.coord - q_center))
            if d <= 10.0 or a.is_metal:
                pocket_rec_atoms.append(a)

        all_render_atoms = pocket_rec_atoms + query.atoms
        all_coords = np.array([a.coord for a in all_render_atoms])
        center = np.mean(all_coords, axis=0)

        # Apply viewing rotation (30 deg azimuth, 25 deg elevation)
        theta_x = math.radians(25)
        theta_y = math.radians(35)
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(theta_x), -math.sin(theta_x)],
            [0, math.sin(theta_x), math.cos(theta_x)],
        ])
        Ry = np.array([
            [math.cos(theta_y), 0, math.sin(theta_y)],
            [0, 1, 0],
            [-math.sin(theta_y), 0, math.cos(theta_y)],
        ])
        Rot = np.dot(Rx, Ry)

        centered_coords = all_coords - center
        rot_coords = np.dot(centered_coords, Rot.T)

        # Scale to canvas
        span = np.max(np.abs(rot_coords[:, :2]))
        scale = min(width * 0.35, height * 0.35) / max(span, 1.0)

        screen_x = width * 0.5 + rot_coords[:, 0] * scale
        screen_y = height * 0.55 - rot_coords[:, 1] * scale
        depth_z = rot_coords[:, 2]

        # Sort by depth (painter's algorithm)
        order = np.argsort(depth_z)

        # Precompute atom 2D projection lookup
        proj_map = {}
        for idx, atom in enumerate(all_render_atoms):
            proj_map[atom.index] = (screen_x[idx], screen_y[idx])

        # 1. Draw interaction dashed rays
        for c in contacts:
            r_pt = proj_map.get(c.receptor_atom_idx)
            q_pt = proj_map.get(c.query_atom_idx)
            if r_pt and q_pt:
                itype = c.interaction_type
                col = COLORS.get(itype, (200, 200, 200))
                _draw_dashed_line(draw, r_pt, q_pt, fill=col, width=2)
                # Label distance
                mx = (r_pt[0] + q_pt[0]) / 2.0
                my = (r_pt[1] + q_pt[1]) / 2.0
                draw.text((mx - 10, my - 6), f"{c.distance:.1f}Å", fill=col, font=font_atom)

        q_indices = {a.index for a in query.atoms}

        # 2. Draw atoms sorted by depth
        for o_idx in order:
            atom = all_render_atoms[o_idx]
            px = screen_x[o_idx]
            py = screen_y[o_idx]
            elem = atom.element.upper()

            # Radius based on element
            if atom.is_metal:
                radius = 18
                fill_color = COLORS.get(elem, (168, 85, 247))
                # Outer glow for metal
                draw.ellipse((px - radius - 5, py - radius - 5, px + radius + 5, py + radius + 5), fill=None, outline=(192, 132, 252), width=2)
            elif atom.index in q_indices:
                radius = 10
                fill_color = COLORS.get(elem, (96, 165, 250))
            else:
                radius = 7
                fill_color = (148, 163, 184)  # Muted receptor

            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=fill_color, outline=(255, 255, 255), width=1)

            # Label every ligand atom (and metals) so the docked structure is
            # fully readable; receptor atoms stay unlabelled to reduce clutter.
            if atom.is_metal:
                draw.text((px + radius + 4, py - 7), elem,
                          fill=(241, 245, 249), font=font_atom)
            elif atom.index in q_indices:
                draw.text((px + radius + 3, py - 6), elem,
                          fill=(226, 232, 240), font=font_atom)

        # Legend panel in dark mode
        leg_box = (width - 320, 30, width - 30, 234)
        _draw_rounded_rect(draw, leg_box, fill=(30, 41, 59), outline=(51, 65, 85), radius=8)
        draw.text((width - 300, 45), "VISUAL KEY", fill=(241, 245, 249), font=font_sub)
        draw.ellipse((width - 300, 75, width - 284, 91), fill=COLORS["ZN"])
        draw.text((width - 275, 75), "Metal Cofactor (Zn²⁺)", fill=(203, 213, 225), font=font_atom)
        draw.ellipse((width - 300, 105, width - 288, 117), fill=COLORS["O"])
        draw.text((width - 275, 103), "Oxygen (O)", fill=(203, 213, 225), font=font_atom)
        draw.ellipse((width - 300, 131, width - 288, 143), fill=COLORS["N"])
        draw.text((width - 275, 129), "Nitrogen (N)", fill=(203, 213, 225), font=font_atom)
        draw.ellipse((width - 300, 157, width - 288, 169), fill=COLORS["C"])
        draw.text((width - 275, 155), "Carbon (C)", fill=(203, 213, 225), font=font_atom)
        draw.ellipse((width - 300, 183, width - 288, 195), fill=(148, 163, 184))
        draw.text((width - 275, 181), "Receptor atom", fill=(203, 213, 225), font=font_atom)

        _draw_dashed_line(draw, (width - 300, 212), (width - 270, 212), fill=COLORS["metal_coordination"], width=3)
        draw.text((width - 260, 205), "Coordination Bond", fill=COLORS["metal_coordination"], font=font_atom)

        img.save(output_png_path, "PNG", dpi=(150, 150))
        return output_png_path
