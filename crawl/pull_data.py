import scrapy

class DataUSA_Spider(scrapy.Spider):
    name = "DataUSA_Spider"
    start_urls = ['http://quotes.toscrape.com/page/1/']

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'quotes-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Save file %s' % filename)

    def start_request(self):
        yield scrapy.Request(url=start_urls[0], callback=self.parse)
