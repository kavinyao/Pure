from dragnet.models import kohlschuetter_model, kohlschuetter_css_weninger_model

from util import str_to_unicode

class L3SModel(object):
    def train(self, documents):
        pass

    def predict(self, document):
        s = kohlschuetter_model.analyze(document.html_string.encode('utf-8'))
        return str_to_unicode(s)

class DragnetModel(object):
    def train(self, documents):
        pass

    def predict(self, document):
        s = kohlschuetter_css_weninger_model.analyze(document.html_string.encode('utf-8'))
        return str_to_unicode(s)
