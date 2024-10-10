# Edge class
class EdgeCommon():
    def __init__(self, idx, element_info):
        # idx: the index of the edge
        self.idx = idx
        # element_info: a list of tuples,
        # each tuple = (element, curve index, mark_align)
        # element: element class
        # curve index=1,2,3,4: the index of the curve in the element
        # mark_align: marker for whether the index along the curve aligns with the curve index
        # In fact, we only need one marker to show whether the node indices along the boundaries
        # of the two elements are the same or oppositive. But for the convenience of coding
        # a mapping function to get the correspondance between the index on the common edge and 
        # those on the two boundaries, we use one marker for each element. 
        # We always set mark_align of the first element be 1. If the other element has reversed 
        # index order, we set its mark_align be -1. If not, we set 1.
        self.element_info = element_info
        
        # Check compatibility and getting the maximum node label on the common edge
        num_boundary_node_list = []
        for e, idx_curve, mark_align in element_info:
            if idx_curve==1 or idx_curve==3:
                num_boundary_node_list.append(e.grid.Nx)
            elif idx_curve==2 or idx_curve==4:
                num_boundary_node_list.append(e.grid.Ny) 
                
        if num_boundary_node_list[0]==num_boundary_node_list[1]:
            #print("Compatible")
            self.N = num_boundary_node_list[0]
        else:
            print("Incompatible!")
            self.N = None
        
        
        
        