"""Build Blender-native visualization grids from committed Alembic Points."""

import bpy

ASSET_NAME = "FluidiTy Point Density and Velocity"
VOXEL_SIZE = 0.125


def supported():
    return (
        bpy.app.version >= (5, 3, 0)
        and bpy.app.build_options.openvdb
        and all(
            hasattr(bpy.types, name)
            for name in ("GeometryNodeRasterizePoints", "GeometryNodeFieldToGrid", "GeometryNodeStoreNamedGrid")
        )
    )


def create_grid_asset():
    """Create a volume asset with point count per m³ and mean velocity in m/s."""
    tree = bpy.data.node_groups.new(ASSET_NAME, "GeometryNodeTree")
    tree.asset_mark()
    tree.use_fake_user = True
    tree.asset_data.description = "Rasterize committed FluidiTy points into number density and mean velocity grids."
    tree.interface.new_socket(name="Volume", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    points = tree.interface.new_socket(name="Points", in_out="INPUT", socket_type="NodeSocketGeometry")
    points.description = "Raw Alembic points with the velocity attribute, before display instancing."
    voxel = tree.interface.new_socket(name="Voxel Size", in_out="INPUT", socket_type="NodeSocketFloat")
    voxel.default_value = VOXEL_SIZE
    voxel.min_value = 0.01
    voxel.max_value = 10.0
    voxel.description = "Voxel edge length in meters in the playback object's local space."

    def node(kind, x, y):
        result = tree.nodes.new(kind)
        result.location = (x, y)
        return result

    link = tree.links.new
    source = node("NodeGroupInput", -850, 120)
    output = node("NodeGroupOutput", 800, 120)
    velocity = node("GeometryNodeInputNamedAttribute", -850, -100)
    velocity.data_type = "FLOAT_VECTOR"
    velocity.inputs["Name"].default_value = "velocity"

    raster = node("GeometryNodeRasterizePoints", -630, 120)
    raster.inputs["Kernel Type"].default_value = "Constant"
    raster.rasterize_items.new("Count")
    raster.rasterize_items.new("Velocity Sum").type = "VECTOR"
    raster.inputs["Count"].default_value = 1.0
    link(source.outputs["Points"], raster.inputs["Points"])
    link(source.outputs["Voxel Size"], raster.inputs["Voxel Size"])
    link(velocity.outputs["Attribute"], raster.inputs["Velocity Sum"])

    count = node("GeometryNodeSampleGrid", -390, 100)
    count.data_type = "FLOAT"
    count.inputs["Interpolation"].default_value = "Nearest Neighbor"
    link(raster.outputs["Count"], count.inputs["Grid"])
    velocity_sum = node("GeometryNodeSampleGrid", -390, -120)
    velocity_sum.data_type = "VECTOR"
    velocity_sum.inputs["Interpolation"].default_value = "Nearest Neighbor"
    link(raster.outputs["Velocity Sum"], velocity_sum.inputs["Grid"])

    voxel_volume = node("ShaderNodeMath", -390, -350)
    voxel_volume.operation = "POWER"
    voxel_volume.inputs[1].default_value = 3.0
    link(source.outputs["Voxel Size"], voxel_volume.inputs[0])
    density = node("ShaderNodeMath", -170, 100)
    density.operation = "DIVIDE"
    link(count.outputs["Value"], density.inputs[0])
    link(voxel_volume.outputs[0], density.inputs[1])
    reciprocal = node("ShaderNodeMath", -170, -90)
    reciprocal.operation = "DIVIDE"
    reciprocal.inputs[0].default_value = 1.0
    link(count.outputs["Value"], reciprocal.inputs[1])
    mean_velocity = node("ShaderNodeVectorMath", 30, -110)
    mean_velocity.operation = "SCALE"
    link(velocity_sum.outputs["Value"], mean_velocity.inputs[0])
    link(reciprocal.outputs[0], mean_velocity.inputs["Scale"])

    # Rasterize Points activates a halo even for the constant kernel.
    empty = node("ShaderNodeMath", -170, 290)
    empty.operation = "LESS_THAN"
    empty.inputs[1].default_value = 0.5
    link(count.outputs["Value"], empty.inputs[0])
    occupied = node("GeometryNodeGridDeactivateVoxels", 30, 290)
    link(raster.outputs["Count"], occupied.inputs["Grid"])
    link(empty.outputs[0], occupied.inputs["Selection"])

    fields = node("GeometryNodeFieldToGrid", 230, 120)
    fields.grid_items.new("FLOAT", "Density")
    fields.grid_items.new("VECTOR", "Velocity")
    link(occupied.outputs["Grid"], fields.inputs["Topology"])
    link(density.outputs[0], fields.inputs["Density"])
    link(mean_velocity.outputs["Vector"], fields.inputs["Velocity"])

    density_grid = node("GeometryNodeStoreNamedGrid", 430, 120)
    density_grid.inputs["Name"].default_value = "density"
    link(fields.outputs["Density"], density_grid.inputs["Grid"])
    velocity_grid = node("GeometryNodeStoreNamedGrid", 620, 120)
    velocity_grid.data_type = "VECTOR_FLOAT"
    velocity_grid.inputs["Name"].default_value = "velocity"
    link(density_grid.outputs["Volume"], velocity_grid.inputs["Volume"])
    link(fields.outputs["Velocity"], velocity_grid.inputs["Grid"])
    link(velocity_grid.outputs["Volume"], output.inputs["Volume"])
    return tree
