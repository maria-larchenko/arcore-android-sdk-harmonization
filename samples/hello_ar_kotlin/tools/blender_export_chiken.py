# blender_export_chiken.py
"""
Usage (from a terminal):

    blender chiken.blend --background --python blender_export_chiken.py -- <output_dir> <base_name>

Example:

    blender chiken.blend --background --python blender_export_chiken.py -- ./export chiken

This script will:
1. Load the provided .blend file.
2. Smart UV-unwrap every mesh object.
3. Bake each object's diffuse albedo into a single 2K texture.
4. Export an OBJ file and its corresponding PNG texture (named
   <base_name>.obj and <base_name>_albedo.png) into <output_dir>.

Tested with Blender 4.x and the Cycles render engine.
"""

import bpy
import os
import sys
from typing import List


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def parse_args(argv: List[str]):
    """Extracts script arguments placed after "--" by Blender."""
    if "--" not in argv:
        raise SystemExit(
            "Missing script arguments. Usage: blender <blend_file> --background "
            "--python blender_export_chiken.py -- <output_dir> <base_name>"
        )

    idx = argv.index("--")
    args = argv[idx + 1 :]

    if len(args) < 2:
        raise SystemExit(
            "Expected 2 arguments <output_dir> <base_name>. Got: {}".format(args)
        )

    output_dir = os.path.abspath(args[0])
    base_name = args[1]

    os.makedirs(output_dir, exist_ok=True)
    return output_dir, base_name


# -----------------------------------------------------------------------------
# Core steps
# -----------------------------------------------------------------------------


def set_cycles_engine():
    """Ensure Cycles render engine is active for baking."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.bake_type = "DIFFUSE"



def uv_unwrap_and_pack(obj):
    """Smart-unwrap the mesh and pack UV islands to avoid overlaps."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    # Apply smart projection for base unwrap
    bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)
    # Pack islands globally to guarantee they fit within 0-1 space without overlap
    bpy.ops.uv.pack_islands(margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")



def join_mesh_objects(mesh_objects):
    """Join all mesh objects into a single object. Returns the joined object."""
    if len(mesh_objects) == 1:
        return mesh_objects[0]

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.object.join()
    return bpy.context.view_layer.objects.active



def ensure_bake_image(img_name: str, output_path: str):
    """Create or retrieve the shared bake image and hook it to materials."""
    img = bpy.data.images.get(img_name)
    if img is None:
        img = bpy.data.images.new(name=img_name, width=2048, height=2048, alpha=False)
    img.filepath_raw = output_path
    img.file_format = "PNG"
    return img



def attach_image_to_materials(obj, img):
    """Attach the bake image to every material of the object and make it active."""
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None:
            continue
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        # Create a new texture node dedicated to baking.
        tex_node = nodes.new(type="ShaderNodeTexImage")
        tex_node.image = img
        # Make this node the active one for baking.
        nodes.active = tex_node



def bake_to_image():
    """Run the diffuse color bake with only the COLOR pass."""
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, use_clear=True)



def export_obj(filepath: str):
    """Export the entire scene (all objects) to OBJ, compatible with Blender 4.x and older.

    Order of attempts:
    1. New operator (Blender ≥4.0): ``bpy.ops.wm.obj_export``.
    2. Legacy operator (Blender <4.0): ``bpy.ops.export_scene.obj``.
    3. Try to enable the OBJ add-on automatically and retry (rare cases).
    """
    # 1) Preferred operator in Blender 4.x
    if hasattr(bpy.ops.wm, "obj_export"):
        try:
            bpy.ops.wm.obj_export(
                filepath=filepath,
                export_selected_objects=False,
                export_materials=False,
                forward_axis="NEGATIVE_Z",
                up_axis="Y",
            )
            return
        except RuntimeError:
            # Fall back if operator is present but add-on disabled.
            pass

    # 2) Legacy operator
    if hasattr(bpy.ops.export_scene, "obj"):
        bpy.ops.export_scene.obj(
            filepath=filepath,
            use_selection=False,
            use_materials=False,
            axis_forward="-Z",
            axis_up="Y",
        )
        return

    # 3) Attempt to enable addon then retry new operator
    try:
        import addon_utils

        addon_utils.enable("io_scene_obj")
        if hasattr(bpy.ops.wm, "obj_export"):
            bpy.ops.wm.obj_export(
                filepath=filepath,
                export_selected_objects=False,
                export_materials=False,
                forward_axis="NEGATIVE_Z",
                up_axis="Y",
            )
            return
    except Exception:
        pass

    # If we reach here, give up.
    raise RuntimeError(
        "OBJ export operator not found. Ensure the OBJ I/O add-on is enabled or install a compatible Blender version."
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    output_dir, base_name = parse_args(sys.argv)

    set_cycles_engine()

    # Prepare a shared image for baking.
    albedo_path = os.path.join(output_dir, f"{base_name}_albedo.png")
    bake_img = ensure_bake_image(f"{base_name}_albedo", albedo_path)

    # Collect mesh objects and join them to ensure unique UV space
    original_mesh_objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]

    joined_obj = join_mesh_objects(original_mesh_objects)

    uv_unwrap_and_pack(joined_obj)

    # Attach bake image to materials of the joined object
    attach_image_to_materials(joined_obj, bake_img)

    # For baking/selecting, work with the single joined object list
    mesh_objects = [joined_obj]

    # Select all mesh objects for baking.
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]

    bake_to_image()
    bake_img.save()

    # Export OBJ.
    obj_path = os.path.join(output_dir, f"{base_name}.obj")
    export_obj(obj_path)

    print(f"Export finished:\n  OBJ: {obj_path}\n  Albedo: {albedo_path}")


if __name__ == "__main__":
    main() 