from tqdm import tqdm
import orjson
import argparse

def make_a_train(input_file, output_file):

# Load the JSON file
    with open(input_file, "rb") as file:
        data = orjson.loads(file.read())
    # Create a set to store seen keys 
    seen_keys = set()
    # Create a new dictionary with the keys from the original JSON and rel_ins_ids as values
    new_dict = {}
    for key, value in tqdm(data["data"].items()):
        if key not in seen_keys:
            try:
                # Check if rel_ins_ids are in the original JSON
                valid_rel_ins_ids = [rel_ins_id for rel_ins_id in value["rel_ins_ids"] if rel_ins_id in data["data"]]
                # Add the valid rel_ins_ids to the new_dict
                new_dict[key] = valid_rel_ins_ids
                seen_keys.update(valid_rel_ins_ids)
                
            except Exception as e:
                print("Error with key %s and value %s" % (key, value))
    # Write the new dictionary to a new JSON file
    with open(output_file, "wb") as file:
        file.write(orjson.dumps(new_dict))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='/data/chengshuang/Otter/output/FunQAf128_instructions_train.json')
    parser.add_argument('--output_file', type=str, default='/data/chengshuang/Otter/output/FunQAf128_train_train.json')
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    make_a_train(input_file, output_file)
    # make_a_train("/mnt/bn/ecom-govern-maxiangqian-lq/lj/data/dwq/FunQAf128_instructions_train.json", "/mnt/bn/ecom-govern-maxiangqian-lq/lj/data/dwq/FunQAf128_train_train.json")