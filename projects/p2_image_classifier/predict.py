from argparse import ArgumentParser

from utilities import load_saved_model, predict, load_json_mapping

def parse_shell_args():
    parser = ArgumentParser()
    parser.add_argument(
        "image",
        help="Image PATH to predict the class",
        type=str
    )
    parser.add_argument(
        "model",
        help="Model PATH used to predict",
        type=str
    )
    parser.add_argument("--category_names",
                        "-n",
                        help="path to a JSON file mapping labels to flower names",
                        type=str,
                        default="label_map.json",
                        required=False,
                        )
    parser.add_argument("--top_k",
                        "-k",
                        help="top K most likely classes",
                        type=int,
                        default=5,
                        required=False,
                        )
    parsed_args = parser.parse_args()
    return (
        parsed_args.image,
        parsed_args.model,
        parsed_args.category_names,
        parsed_args.top_k
    )

if __name__ == '__main__':
    image_path, model_path, mapping_file_path, top_n = parse_shell_args()
    model = load_saved_model(model_path)
    mapping = load_json_mapping(mapping_file_path)
    top_predicted_prob, top_indexes = predict(image_path, model, top_n)
    print("Results:")
    print("-" * 12)
    for prob, map_index in zip(top_predicted_prob, top_indexes):
        class_name = mapping[map_index]
        print("Class name: {} probability: {}".format(class_name, str(prob)))
    print("-" * 12)
