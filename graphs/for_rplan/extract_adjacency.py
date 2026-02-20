from shapely.geometry import Polygon

def are_adjacent(poly_a: Polygon,poly_b:Polygon,tol=1e-3):
    return poly_a.buffer(tol).intersects(poly_b.buffer(tol))
def spatial_direction(ca, cb):
    dx = cb[0] - ca[0]
    dy = cb[1] - ca[1]

    if abs(dx) > abs(dy):
        return "right_of" if dx > 0 else "left_of"
    else:
        return "below" if dy > 0 else "above"