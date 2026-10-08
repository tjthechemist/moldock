"""
Visual analysis and PNG rendering for molecular docking results:
2D Interaction Networks, 3D Binding Site Projections, and Multi-Panel Summary Dashboards.
"""

import math
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moldock.core.atom import Atom
from moldock.core.molecule import Molecule
from moldock.metals.detector import MetalCenter
from moldock.analysis.interactions import InteractionContact, InteractionProfiler


# Professional CPK and Interaction Color Palette
COLORS = {
    "bg": (255, 255, 255),
    "panel_bg": (248, 250, 252),
    "border": (226, 232, 240),
    "text_dark": (30, 41, 59),
    "text_muted": (100, 116, 139),
    # Elements
    "C": (74, 85, 104),
    "N": (49, 130, 206),
    "O": (229, 62, 62),
    "S": (214, 158, 46),
    "H": (160, 174, 192),
    "P": (237, 137, 54),
    "CL": (56, 161, 105),
    "F": (72, 187, 120),
    "BR": (155, 44, 44),
    # Metals
    "ZN": (128, 90, 213),
    "MG": (107, 70, 193),
    "CA": (142, 68, 173),
    "FE": (197, 48, 48),
    "CU": (49, 151, 149),
    "MN": (159, 122, 234),
    # Interactions
    "metal_coordination": (159, 122, 234),  # Vivid Purple
    "hbond": (56, 161, 105),               # Emerald Green
    "salt_bridge": (221, 107, 32),          # Amber Orange
    "hydrophobic": (113, 128, 150),         # Slate Gray
}


def _get_font(size: int = 14) -> ImageFont.ImageFont:
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
        Generate a 2D interaction network diagram.
        The docked ligand/partner is centered, surrounded by coordinating metal ions
        and interacting protein residues with color-coded interaction links and distances.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
        img = Image.new("RGB", (width, height), COLORS["bg"])
        draw = ImageDraw.Draw(img)

        font_title = _get_font(22)
        font_sub = _get_font(13)
        font_node = _get_font(12)
        font_dist = _get_font(11)

        # 1. Header and background banner
        _draw_rounded_rect(draw, (20, 16, width - 20, 75), fill=COLORS["panel_bg"], outline=COLORS["border"], radius=10)
        draw.text((40, 26), title, fill=COLORS["text_dark"], font=font_title)

        subtitle_parts = [f"Receptor: {receptor.name}", f"Query: {query.name}", f"Interactions: {len(contacts)}"]
        if score_info and "total_score" in score_info:
            subtitle_parts.append(f"Affinity: {score_info['total_score']:.2f} kcal/mol")
        if score_info and "metal_score" in score_info and abs(score_info["metal_score"]) > 1e-3:
            subtitle_parts.append(f"Metal Score: {score_info['metal_score']:.2f} kcal/mol")

        draw.text((40, 52), "  |  ".join(subtitle_parts), fill=COLORS["text_muted"], font=font_sub)

        # 2. Layout regions:
        # Left/Center: Interaction Graph (center: x=500, y=480)
        # Right: Interaction Table and Details Panel
        cx, cy = 520, 480
        center_radius = 120.0
        orbit_radius = 310.0

        # Unique interacting query atoms
        q_atoms_map = {a.index: a for a in query.atoms}
        interacting_q_indices = sorted(list(set(c.query_atom_idx for c in contacts)))

        # Unique interacting receptor residues and metals
        rec_partners: Dict[str, Dict[str, Any]] = {}
        for c in contacts:
            key = f"{c.receptor_res}_{c.receptor_res_seq}"
            if key not in rec_partners:
                # Check if metal
                r_atom = next((a for a in receptor.atoms if a.index == c.receptor_atom_idx), None)
                is_metal = r_atom.is_metal if r_atom else (c.interaction_type == "metal_coordination")
                rec_partners[key] = {
                    "label": f"{c.receptor_res}{c.receptor_res_seq}",
                    "is_metal": is_metal,
                    "element": r_atom.element.upper() if r_atom else "M",
                    "contacts": [],
                }
            rec_partners[key]["contacts"].append(c)

        # Place Query atoms in inner circle
        num_q = max(len(interacting_q_indices), 1)
        q_positions: Dict[int, Tuple[float, float]] = {}
        for i, q_idx in enumerate(interacting_q_indices):
            angle = 2.0 * math.pi * i / num_q - (math.pi / 2.0)
            qx = cx + center_radius * math.cos(angle)
            qy = cy + center_radius * math.sin(angle)
            q_positions[q_idx] = (qx, qy)

        # Draw Ligand / Query Core boundary
        draw.ellipse(
            (cx - center_radius - 35, cy - center_radius - 35, cx + center_radius + 35, cy + center_radius + 35),
            outline=(203, 213, 225),
            width=2,
        )
        draw.text((cx - 30, cy - 8), "DOCKED CORE", fill=COLORS["text_muted"], font=font_sub)

        # Place Receptor partners in outer orbit
        r_keys = list(rec_partners.keys())
        num_r = max(len(r_keys), 1)
        r_positions: Dict[str, Tuple[float, float]] = {}
        for i, key in enumerate(r_keys):
            angle = 2.0 * math.pi * i / num_r - (math.pi / 2.0)
            rx = cx + orbit_radius * math.cos(angle)
            ry = cy + orbit_radius * math.sin(angle)
            r_positions[key] = (rx, ry)

        # 3. Draw interaction connecting lines
        for c in contacts:
            q_pos = q_positions.get(c.query_atom_idx)
            rec_key = f"{c.receptor_res}_{c.receptor_res_seq}"
            r_pos = r_positions.get(rec_key)
            if not q_pos or not r_pos:
                continue

            itype = c.interaction_type
            color = COLORS.get(itype, COLORS["text_muted"])

            # Metal coordination drawn as solid thick line; others dashed
            if itype == "metal_coordination":
                draw.line([r_pos, q_pos], fill=color, width=4)
            else:
                _draw_dashed_line(draw, r_pos, q_pos, fill=color, width=2)

            # Distance tag at midpoint
            mx = (r_pos[0] + q_pos[0]) / 2.0
            my = (r_pos[1] + q_pos[1]) / 2.0
            dist_text = f"{c.distance:.1f}Å"
            _draw_rounded_rect(draw, (int(mx - 18), int(my - 9), int(mx + 18), int(my + 9)), fill=(255, 255, 255), outline=color, radius=4)
            draw.text((mx - 14, my - 6), dist_text, fill=color, font=font_dist)

        # 4. Draw Query Atom Nodes
        for q_idx, pos in q_positions.items():
            atom = q_atoms_map.get(q_idx)
            elem = atom.element.upper() if atom else "C"
            elem_color = COLORS.get(elem, COLORS["C"])

            draw.ellipse((pos[0] - 16, pos[1] - 16, pos[0] + 16, pos[1] + 16), fill=elem_color, outline=(255, 255, 255), width=2)
            name_label = atom.name if atom else str(q_idx)
            draw.text((pos[0] - 10, pos[1] - 7), name_label[:3], fill=(255, 255, 255), font=font_node)

        # 5. Draw Receptor Nodes
        for key, pos in r_positions.items():
            data = rec_partners[key]
            is_metal = data["is_metal"]

            if is_metal:
                # Glowing metal badge
                draw.ellipse((pos[0] - 28, pos[1] - 28, pos[0] + 28, pos[1] + 28), fill=(243, 232, 255), outline=COLORS["ZN"], width=3)
                draw.ellipse((pos[0] - 22, pos[1] - 22, pos[0] + 22, pos[1] + 22), fill=COLORS["ZN"], outline=(255, 255, 255), width=2)
                draw.text((pos[0] - 14, pos[1] - 7), data["element"][:2], fill=(255, 255, 255), font=font_node)
                draw.text((pos[0] - 22, pos[1] + 28), f"{data['label']} (Cofactor)", fill=COLORS["ZN"], font=font_sub)
            else:
                # Standard amino acid residue node
                _draw_rounded_rect(
                    draw,
                    (int(pos[0] - 38), int(pos[1] - 18), int(pos[0] + 38), int(pos[1] + 18)),
                    fill=COLORS["panel_bg"],
                    outline=COLORS["border"],
                    radius=8,
                    width=2,
                )
                draw.text((pos[0] - 28, pos[1] - 8), data["label"], fill=COLORS["text_dark"], font=font_node)

        # 6. Right Side: Interaction Details Panel
        panel_x = 960
        panel_w = 400
        _draw_rounded_rect(draw, (panel_x, 100, panel_x + panel_w, height - 30), fill=COLORS["panel_bg"], outline=COLORS["border"], radius=12)

        draw.text((panel_x + 20, 115), "INTERACTION CONTACTS", fill=COLORS["text_dark"], font=font_title)
        draw.line([(panel_x + 20, 148), (panel_x + panel_w - 20, 148)], fill=COLORS["border"], width=1)

        # Legend
        leg_y = 160
        legends = [
            ("Metal Coordination", COLORS["metal_coordination"], True),
            ("Hydrogen Bond", COLORS["hbond"], False),
            ("Salt Bridge", COLORS["salt_bridge"], False),
            ("Hydrophobic Contact", COLORS["hydrophobic"], False),
        ]
        for leg_text, col, is_solid in legends:
            if is_solid:
                draw.line([(panel_x + 22, leg_y + 8), (panel_x + 60, leg_y + 8)], fill=col, width=4)
            else:
                _draw_dashed_line(draw, (panel_x + 22, leg_y + 8), (panel_x + 60, leg_y + 8), fill=col, width=2)
            draw.text((panel_x + 70, leg_y), leg_text, fill=COLORS["text_dark"], font=font_sub)
            leg_y += 26

        draw.line([(panel_x + 20, leg_y + 10), (panel_x + panel_w - 20, leg_y + 10)], fill=COLORS["border"], width=1)
        row_y = leg_y + 25

        # Table header
        draw.text((panel_x + 20, row_y), "Type", fill=COLORS["text_muted"], font=font_dist)
        draw.text((panel_x + 130, row_y), "Receptor", fill=COLORS["text_muted"], font=font_dist)
        draw.text((panel_x + 235, row_y), "Ligand", fill=COLORS["text_muted"], font=font_dist)
        draw.text((panel_x + 325, row_y), "Dist (Å)", fill=COLORS["text_muted"], font=font_dist)
        row_y += 20

        # List contact items (up to 18 rows)
        for c in contacts[:18]:
            type_short = c.interaction_type.replace("_coordination", "").replace("_", " ").upper()
            type_col = COLORS.get(c.interaction_type, COLORS["text_dark"])
            draw.text((panel_x + 20, row_y), type_short[:9], fill=type_col, font=font_dist)
            rec_str = f"{c.receptor_res}{c.receptor_res_seq}:{c.receptor_atom_name}"
            draw.text((panel_x + 130, row_y), rec_str[:12], fill=COLORS["text_dark"], font=font_dist)
            lig_str = f"{c.query_atom_name} ({c.query_element})"
            draw.text((panel_x + 235, row_y), lig_str[:12], fill=COLORS["text_dark"], font=font_dist)
            draw.text((panel_x + 330, row_y), f"{c.distance:.2f}", fill=COLORS["text_dark"], font=font_dist)
            row_y += 22

        if len(contacts) > 18:
            draw.text((panel_x + 20, row_y + 5), f"... and {len(contacts) - 18} more contacts", fill=COLORS["text_muted"], font=font_dist)

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

            if atom.is_metal or atom.name in ("O1", "O2", "N1", "S1", "ZN"):
                draw.text((px + radius + 3, py - 6), f"{atom.name}", fill=(241, 245, 249), font=font_atom)

        # Legend panel in dark mode
        leg_box = (width - 320, 30, width - 30, 210)
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

        _draw_dashed_line(draw, (width - 300, 188), (width - 270, 188), fill=COLORS["metal_coordination"], width=3)
        draw.text((width - 260, 181), "Coordination Bond", fill=COLORS["metal_coordination"], font=font_atom)

        img.save(output_png_path, "PNG", dpi=(150, 150))
        return output_png_path
