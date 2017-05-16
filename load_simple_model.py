import argparse 
import tensorflow as tf


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
#    for op in graph.get_operations():
#        print(op.name)
        
    # We access the input and output nodes 
    x = graph.get_tensor_by_name('prefix/Model/v1:0')
    y = graph.get_tensor_by_name('prefix/Model/v2:0')
    a= graph.get_tensor_by_name('prefix/Model/add:0')



    # We launch a Session
    with tf.Session(graph=graph) as sess:
        
        print(sess.run(a, feed_dict={x: 1, y:6}))
        print("Loading is successful!")