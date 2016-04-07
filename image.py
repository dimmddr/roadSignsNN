class Image(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.signs = []

    def add_sign(self, label, coordinates):
        self.signs.append(Sign(label, coordinates))


class Sign(object):
    def __init__(self, label, coordinates):
        self.label = label
        self.coord = coordinates
