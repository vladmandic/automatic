from modules import shared


class Detailer:
    def name(self):
        return "None"

    def restore(self, np_image):
        return np_image


def detail(np_image, p=None):
    detailers = [x for x in shared.detailers if x.name() == shared.opts.detailer_model or shared.opts.detailer_model is None]
    if len(detailers) == 0:
        return np_image
    detailer = detailers[0]
    return detailer.restore(np_image, p)
