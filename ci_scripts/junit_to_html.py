from collections import defaultdict, namedtuple
from jinja2 import Template
from xml.etree import ElementTree as ET
import sys

def parse_junit_xml(xml_files):
    """
    Parses JUnit XML file and organizes test cases by label and test suite.
    """
    MyTestSuite = namedtuple('MyTestSuite', ['labels', 'data', 'test_cases'])
    # test_suites = defaultdict(lambda: MyTestSuite({}, []))
    test_suites = []

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for testsuite in root.findall('testsuite'):
            labels = testsuite.attrib['name'].split('/')
            testsuite_data = {
                'tests': testsuite.attrib['tests'],
                'failures': testsuite.attrib['failures'],
                'errors': testsuite.attrib['errors'],
                'time': testsuite.attrib['time'],
                'id': testsuite.attrib['id'],
            }
            testsuite_cases = []
            for testcase in testsuite.findall('testcase'):
                name = testcase.attrib['name']
                time = testcase.attrib['time']
                failure = None
                failure_elem = testcase.find('failure')
                if failure_elem is not None:
                    failure = {
                        'type': failure_elem.attrib.get('type', ''),
                        'message': failure_elem.attrib.get('message', ''),
                        'text': failure_elem.text.strip()
                    }
                testsuite_cases.append({
                    'name': name,
                    'time': time,
                    'failure': failure
                })

            for failure_elem in testsuite.findall('error'):
                name = ""
                time = failure_elem.attrib['time']
                failure = {
                    'type': failure_elem.attrib.get('type', ''),
                    'message': failure_elem.attrib.get('message', ''),
                    'text': failure_elem.text.strip()
                }
                testsuite_cases.append({
                    'name': name,
                    'time': time,
                    'failure': failure
                })


            ts = MyTestSuite(labels, testsuite_data, testsuite_cases)
            test_suites.append(ts)
    
    return test_suites

def generate_html_report(test_suites, output_file):
    """
    Generates HTML report using Jinja2 template.
    """
    with open('ci_scripts/junit_template.html', 'r') as f:
        template_content = f.read()

    template = Template(template_content)
    test_suites_sorted = sorted(test_suites, key=lambda ts: ts.labels)
    html_content = template.render(test_suites=test_suites)

    with open(output_file, 'w') as f:
        f.write(html_content)

if __name__ == '__main__':
    html_output_file = sys.argv[1]
    junit_xml_files = sys.argv[2:]

    test_suites = parse_junit_xml(junit_xml_files)
    generate_html_report(test_suites, html_output_file)
