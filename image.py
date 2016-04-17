class Image(object):
    def __init__(self, filename):
        self.filename = filename
        self.signs = []

    def add_sign(self, label, coordinates):
        self.signs.append(Sign(label, coordinates))

    def get_coordinates(self):
        res = []
        for sign in self.signs:
            res.append(sign.coord)
        return res


class Sign(object):
    def __init__(self, label, coordinates):
        self.label = label
        self.coord = coordinates

    def __repr__(self):
        return "Sign label: {}, \n\tSign coordinates: {}\n".format(self.label, self.coord)
