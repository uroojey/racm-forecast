import networkx as nx
from pyvis.network import Network
from .schema import NodeType

class GraphVisualizer:
    """Visualizes the networkx graph using pyvis."""
    
    COLOR_MAP = {
        NodeType.COMPANY.value: "#1f78b4",   # Blue
        NodeType.SECTOR.value: "#33a02c",    # Green
        NodeType.EVENT.value: "#e31a1c",     # Red
        NodeType.MACRO.value: "#984ea3"      # Purple
    }

    def __init__(self, notebook: bool = False):
        self.notebook = notebook

    def visualize(self, graph: nx.MultiDiGraph, output_path: str = "graph.html") -> None:
        """
        Converts the networkx graph to a pyvis Network.
        Apply colors based on the 'node_type' attribute.
        Save the HTML file to output_path.
        """
        net = Network(notebook=self.notebook, directed=True)
        
        for node, attrs in graph.nodes(data=True):
            node_type = attrs.get('node_type', 'unknown')
            color = self.COLOR_MAP.get(node_type, "#999999")
            label = attrs.get('name', str(node))
            net.add_node(node, label=label, color=color, title=str(attrs))
            
        for source, target, attrs in graph.edges(data=True):
            edge_type = attrs.get('edge_type', '')
            net.add_edge(source, target, title=edge_type)
            
        if self.notebook:
            net.show(output_path)
        else:
            net.save_graph(output_path)
