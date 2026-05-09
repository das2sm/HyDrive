import xml.etree.ElementTree as ET
import copy

def repeat_route(base_route, repeat_num):
    tree = ET.parse(f'{base_route}')
    root = tree.getroot()
    cases = root.findall('route')
    new_root = ET.Element("routes")
    for case in cases:
        for i in range(repeat_num):
            case_i = copy.deepcopy(case)
            case_i.attrib["id"] = case_i.attrib["id"] + f"_rp{i}"
            new_root.append(case_i)
    new_tree = ET.ElementTree(new_root)
    # new_tree.write(f'{base_route.split(".")[0]}_rp{repeat_num}.xml', encoding='utf-8', xml_declaration=True)
    return new_tree
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("base_route", type=str)
    parser.add_argument("repeat_num", type=int)
    args = parser.parse_args()
    repeat_route(args.base_route, args.repeat_num)