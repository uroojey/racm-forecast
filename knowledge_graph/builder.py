import networkx as nx
from typing import List, Dict, Optional
from .schema import NodeType, EdgeType
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

class DynamicGraphBuilder:
    """
    Builds and manages a time-evolving financial knowledge graph.
    Maintains a core static graph (companies, sectors) and allows 
    adding temporal events/macro factors to create daily snapshots.
    """
    def __init__(self):
        self.base_graph = nx.MultiDiGraph()
        self.snapshots: Dict[str, nx.MultiDiGraph] = {}

    def add_company(self, ticker: str, name: str, sector: Optional[str] = None) -> None:
        """Adds a company node and optionally connects it to a sector."""
        if not self.base_graph.has_node(ticker):
            self.base_graph.add_node(ticker, name=name, node_type=NodeType.COMPANY.value)
        
        if sector:
            if not self.base_graph.has_node(sector):
                self.base_graph.add_node(sector, name=sector, node_type=NodeType.SECTOR.value)
            self.base_graph.add_edge(ticker, sector, edge_type=EdgeType.BELONGS_TO.value)

    def add_competitor_edge(self, ticker1: str, ticker2: str) -> None:
        """Adds a COMPETES_WITH edge between two companies in the base graph."""
        if self.base_graph.has_node(ticker1) and self.base_graph.has_node(ticker2):
            self.base_graph.add_edge(ticker1, ticker2, edge_type=EdgeType.COMPETES_WITH.value)
        else:
            logger.warning(f"Cannot add competitor edge between {ticker1} and {ticker2}: one or both nodes missing.")

    def build_snapshot(self, date: str, events: List[dict], macro_data: Optional[dict] = None) -> nx.MultiDiGraph:
        """
        Creates a new graph snapshot for a specific date by copying the base_graph
        and adding temporal nodes (Events, Macro).
        
        `events` is a list of dicts (from FinancialEvent).
        Extract event nodes and HAS_EVENT / AFFECTS edges.
        Store the snapshot in self.snapshots[date] and return it.
        """
        snapshot = deepcopy(self.base_graph)
        
        for idx, event in enumerate(events):
            event_id = f"event_{date}_{idx}"
                
            snapshot.add_node(event_id, **event, node_type=NodeType.EVENT.value)
            
            ticker = event.get('ticker')
            if ticker:
                if not snapshot.has_node(ticker):
                    snapshot.add_node(ticker, name=event.get('company', ticker), node_type=NodeType.COMPANY.value)
                snapshot.add_edge(ticker, event_id, edge_type=EdgeType.HAS_EVENT.value)
                snapshot.add_edge(event_id, ticker, edge_type=EdgeType.AFFECTS.value)
                    
        if macro_data:
            for factor, data in macro_data.items():
                node_id = f"macro_{factor}"
                snapshot.add_node(node_id, **data, node_type=NodeType.MACRO.value)

        self.snapshots[date] = snapshot
        return snapshot
    
    def get_snapshot(self, date: str) -> Optional[nx.MultiDiGraph]:
        """Retrieves a previously built snapshot for the given date."""
        return self.snapshots.get(date)
