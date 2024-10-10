# Edge class

class Edge():
    
    def __init__(self, e, idx_edge):
        # The element which the edge belongs to
        self.e = e
        # The edge index within this element, values in 1,2,3,4
        self.idx_edge = idx_edge
        
        # The end nodes of the edge
        # self.c1, self.c2: nodes arranged in the positive edge direction
        self.set_EndNode()
        
        
    def set_EndNode(self):
        # Get the end nodes of the edge
        # The nodes are ordered according to the convention of the quadrilateral
            # edge 1: x1->x2
            # edge 2: x2->x3
            # edge 3: x4->x3
            # edge 4: x1->x4
        # The order is important because it determines whether two coinciding edges
        # are aligned or in the opposite direction.
        e = self.e
        
        if self.idx_edge == 1:
            self.c1 = e.corners[1-1]
            self.c2 = e.corners[2-1]
        elif self.idx_edge == 2:
            self.c1 = e.corners[2-1]
            self.c2 = e.corners[3-1]
        elif self.idx_edge == 3:
            self.c1 = e.corners[4-1]
            self.c2 = e.corners[3-1]
        elif self.idx_edge == 4:
            self.c1 = e.corners[1-1]
            self.c2 = e.corners[4-1]