import gmsh
import pyvista as pv
import os

class BeamFactory:
    def __init__(self, mesh_size=4.0):
        self.mesh_size = mesh_size
        self.filename = "current_beam.vtk"

    def _finalize_mesh(self):
        """Generates mesh and saves it. Session remains open for node counting."""
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_size)
        gmsh.model.mesh.generate(3)
        gmsh.write(self.filename)
        # print(f"Mesh saved to {self.filename}") # Silent mode for 1000 samples

    def get_node_count(self):
        """Extracts node count and then closes the Gmsh session."""
        node_tags, _, _ = gmsh.model.mesh.get_nodes()
        count = len(node_tags)
        gmsh.finalize() 
        return count

    def create_i_beam(self, L, w, h, t_f, t_w):
        gmsh.initialize(); gmsh.model.add("IBeam")
        hw, hh, htw = w/2, h/2, t_w/2
        points = [(-hw, -hh), (hw, -hh), (hw, -hh + t_f), (htw, -hh + t_f), 
                  (htw, hh - t_f), (hw, hh - t_f), (hw, hh), (-hw, hh), 
                  (-hw, hh - t_f), (-htw, hh - t_f), (-htw, -hh + t_f), (-hw, -hh + t_f)]
        p_tags = [gmsh.model.occ.addPoint(p[0], p[1], 0) for p in points]
        l_tags = [gmsh.model.occ.addLine(p_tags[i], p_tags[(i+1)%len(p_tags)]) for i in range(len(p_tags))]
        loop = gmsh.model.occ.addCurveLoop(l_tags)
        surf = gmsh.model.occ.addPlaneSurface([loop])
        gmsh.model.occ.extrude([(2, surf)], 0, 0, L)
        self._finalize_mesh()

    def create_t_beam(self, L, w, h, t_f, t_w):
        gmsh.initialize(); gmsh.model.add("TBeam")
        hw, hh, htw = w/2, h/2, t_w/2
        points = [(-htw, -hh), (htw, -hh), (htw, hh - t_f), (hw, hh - t_f),
                  (hw, hh), (-hw, hh), (-hw, hh - t_f), (-htw, hh - t_f)]
        p_tags = [gmsh.model.occ.addPoint(p[0], p[1], 0) for p in points]
        l_tags = [gmsh.model.occ.addLine(p_tags[i], p_tags[(i+1)%len(p_tags)]) for i in range(len(p_tags))]
        loop = gmsh.model.occ.addCurveLoop(l_tags)
        surf = gmsh.model.occ.addPlaneSurface([loop])
        gmsh.model.occ.extrude([(2, surf)], 0, 0, L)
        self._finalize_mesh()

    def create_hollow_box(self, L, w, h, t):
        gmsh.initialize(); gmsh.model.add("Box")
        outer = gmsh.model.occ.addRectangle(-w/2, -h/2, 0, w, h)
        inner = gmsh.model.occ.addRectangle(-(w-2*t)/2, -(h-2*t)/2, 0, w-2*t, h-2*t)
        surf, _ = gmsh.model.occ.cut([(2, outer)], [(2, inner)])
        gmsh.model.occ.extrude(surf, 0, 0, L)
        self._finalize_mesh()

    def create_circular_tube(self, L, r_out, r_in):
        gmsh.initialize(); gmsh.model.add("Tube")
        c1 = gmsh.model.occ.addDisk(0, 0, 0, r_out, r_out)
        c2 = gmsh.model.occ.addDisk(0, 0, 0, r_in, r_in)
        surf, _ = gmsh.model.occ.cut([(2, c1)], [(2, c2)])
        gmsh.model.occ.extrude(surf, 0, 0, L)
        self._finalize_mesh()