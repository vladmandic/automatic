import time
from modules import shared, ui_extra_networks


class ExtraNetworksPageHistory(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('History')
        # shared.log.trace('History init')
        self.last_refresh = 0

    def refresh(self):
        # shared.log.trace('History refresh')
        self.last_refresh = time.time()
        self.html = '<h1>buttons</h1>'
        for ts in shared.history.list:
            self.html += '<p>' + str(ts) + '</p>'

    def list_items(self):
        # shared.log.trace('History list')
        return shared.history.list

    def create_page(self, tabname, skip = False):
        # shared.log.trace(f'History page: tab={tabname} skip={skip}')
        self.page_time = time.time()
        if tabname == 'txt2img':
            self.last_refresh = time.time()
        if self.page_time <= self.last_refresh: # cached page
            self.refresh()
        return self.patch(self.html, tabname)
