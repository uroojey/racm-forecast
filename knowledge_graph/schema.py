from enum import Enum

class NodeType(str, Enum):
    COMPANY = "company"
    SECTOR = "sector"
    EVENT = "event"
    MACRO = "macro"

class EdgeType(str, Enum):
    BELONGS_TO = "belongs_to"          # Company -> Sector
    COMPETES_WITH = "competes_with"    # Company -> Company
    HAS_EVENT = "has_event"            # Company -> Event
    AFFECTS = "affects"                # Event/Macro -> Company
    CORRELATED_WITH = "correlated_with" # Macro -> Macro
