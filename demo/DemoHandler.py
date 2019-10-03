"""Module for to demo python project
"""
import requests


class DemoHandler(object):
    """Placeholder class"""

    def __init__(self):
        print('access_token=XXXXX')
        print('Hallo, Welt.')

    def do_nothing(self):
        return

    def do_something(self):
        return 'Ain\'t that something?'

    def some_bool(self):
        return True

    def remote_data(self):
        # import pdb; pdb.set_trace();
        response = requests.get('http://httpbin.org/get')
        return response.text


def main():

    demo = DemoHandler()
    print(demo.do_something())
    print(demo.remote_data())


if __name__ == '__main__':

    main()