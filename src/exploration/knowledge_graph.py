"""
Knowledge Graph Generation and Exploration Module.

This module implements sophisticated algorithms for extracting entities and
relationships from documents to build navigable knowledge graphs. It provides
comprehensive graph analysis capabilities and interactive exploration features.

Performance characteristics:
- Optimized for large document collections
- Memory-efficient sparse graph representation
- Multi-perspective graph traversal algorithms
- Hierarchical entity clustering for complexity management
"""

import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Set
import networkx as nx
import spacy
from collections import Counter, defaultdict
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Create and manage knowledge graphs from document collections.
    
    This class implements sophisticated entity and relationship extraction
    to build interconnected knowledge graphs that can be explored and analyzed.
    """
    
    def __init__(self, 
                 language_model: str = "en_core_web_sm",
                 min_entity_freq: int = 2,
                 min_relationship_weight: int = 1,
                 max_entities: int = 1000):
        """
        Initialize the knowledge graph generator with configurable parameters.
        
        Args:
            language_model: spaCy language model to use for NLP
            min_entity_freq: Minimum frequency for entities to be included
            min_relationship_weight: Minimum weight for relationships to be included
            max_entities: Maximum number of entities to include in the graph
        """
        self.min_entity_freq = min_entity_freq
        self.min_relationship_weight = min_relationship_weight
        self.max_entities = max_entities
        
        # Initialize NLP model
        try:
            self.nlp = spacy.load(language_model)
            logger.info(f"Loaded spaCy model: {language_model}")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            logger.error("Run: python -m spacy download en_core_web_sm")
            raise
        
        # Initialize graph
        self.graph = nx.DiGraph()
        self.entity_counts = Counter()
        self.documents_processed = 0
        
        # Entity and relationship types
        self.entity_types = set()
        self.relationship_types = set()
        
        # Performance tracking
        self.extraction_times = []
        self.total_entities_found = 0
        self.total_relationships_found = 0
    
    def process_document(self, 
                         document: Union[str, Dict[str, Any]], 
                         document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document and extract entities and relationships.
        
        This method analyzes document text to identify named entities, concepts,
        and their relationships for integration into the knowledge graph.
        
        Args:
            document: Document text or dictionary with text and metadata
            document_id: Optional identifier for the document
            
        Returns:
            Dictionary with extraction statistics and status
        """
        result = {
            'success': False,
            'document_id': document_id,
            'entities_found': 0,
            'relationships_found': 0,
            'processing_time': 0,
            'error': None
        }
        
        start_time = time.time()
        
        try:
            # Extract text from document
            doc_text = ""
            doc_metadata = {}
            
            if isinstance(document, str):
                doc_text = document
                if not document_id:
                    document_id = f"doc_{self.documents_processed + 1}"
            elif isinstance(document, dict):
                doc_text = document.get('text', '')
                doc_metadata = document.get('metadata', {})
                if not document_id:
                    document_id = doc_metadata.get('id', f"doc_{self.documents_processed + 1}")
            else:
                result['error'] = "Invalid document format. Expected string or dictionary."
                return result
            
            # Process the document with spaCy
            doc = self.nlp(doc_text)
            
            # Extract entities
            entities = []
            for ent in doc.ents:
                entity = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                }
                entities.append(entity)
                
                # Track entity type
                self.entity_types.add(ent.label_)
                
                # Update entity count
                self.entity_counts[ent.text] += 1
            
            # Add document node to graph
            self.graph.add_node(
                document_id,
                type='document',
                metadata=doc_metadata
            )
            
            # Add entity nodes and relationships to graph
            relationships = []
            
            for entity in entities:
                entity_id = f"{entity['text']}_{entity['label']}"
                
                # Add or update entity node
                if not self.graph.has_node(entity_id):
                    self.graph.add_node(
                        entity_id,
                        type='entity',
                        text=entity['text'],
                        label=entity['label'],
                        count=1
                    )
                else:
                    self.graph.nodes[entity_id]['count'] += 1
                
                # Add relationship from document to entity
                rel_key = (document_id, entity_id)
                if self.graph.has_edge(*rel_key):
                    self.graph[document_id][entity_id]['weight'] += 1
                else:
                    self.graph.add_edge(
                        document_id,
                        entity_id,
                        relationship='contains',
                        weight=1
                    )
                    
                    # Track relationship type
                    self.relationship_types.add('contains')
                    
                    # Add to relationships list
                    relationships.append({
                        'source': document_id,
                        'target': entity_id,
                        'relationship': 'contains',
                        'weight': 1
                    })
            
            # Extract entity-entity relationships
            entity_pairs = self._extract_entity_relationships(doc, entities)
            
            for source, target, relationship in entity_pairs:
                source_id = f"{source['text']}_{source['label']}"
                target_id = f"{target['text']}_{target['label']}"
                
                # Add relationship between entities
                rel_key = (source_id, target_id)
                if self.graph.has_edge(*rel_key):
                    self.graph[source_id][target_id]['weight'] += 1
                else:
                    self.graph.add_edge(
                        source_id,
                        target_id,
                        relationship=relationship,
                        weight=1
                    )
                    
                    # Track relationship type
                    self.relationship_types.add(relationship)
                    
                    # Add to relationships list
                    relationships.append({
                        'source': source_id,
                        'target': target_id,
                        'relationship': relationship,
                        'weight': 1
                    })
            
            # Update statistics
            self.documents_processed += 1
            self.total_entities_found += len(entities)
            self.total_relationships_found += len(relationships)
            
            result['entities_found'] = len(entities)
            result['relationships_found'] = len(relationships)
            result['success'] = True
            
        except Exception as e:
            logger.error(f"Error processing document for knowledge graph: {e}")
            result['error'] = str(e)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.extraction_times.append(processing_time)
        result['processing_time'] = processing_time
        
        return result
    
    def process_documents(self, documents: List[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Process multiple documents and update the knowledge graph.
        
        Args:
            documents: List of document texts or dictionaries
            
        Returns:
            Dictionary with overall processing statistics
        """
        result = {
            'success': False,
            'documents_processed': 0,
            'total_entities_found': 0,
            'total_relationships_found': 0,
            'total_processing_time': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            for i, doc in enumerate(documents):
                doc_result = self.process_document(doc, document_id=f"doc_{self.documents_processed + i}")
                
                if doc_result['success']:
                    result['documents_processed'] += 1
                    result['total_entities_found'] += doc_result['entities_found']
                    result['total_relationships_found'] += doc_result['relationships_found']
                else:
                    result['errors'].append(doc_result['error'])
            
            result['success'] = True
            
        except Exception as e:
            logger.error(f"Error processing documents for knowledge graph: {e}")
            result['errors'].append(str(e))
        
        # Calculate total processing time
        result['total_processing_time'] = time.time() - start_time
        
        return result
    
    def prune_graph(self, 
                   min_entity_freq: Optional[int] = None, 
                   min_relationship_weight: Optional[int] = None,
                   max_entities: Optional[int] = None) -> Dict[str, Any]:
        """
        Prune the knowledge graph to improve quality and performance.
        
        This method removes low-frequency entities and weak relationships
        to create a more focused and meaningful knowledge graph.
        
        Args:
            min_entity_freq: Minimum entity frequency (defaults to instance value)
            min_relationship_weight: Minimum relationship weight (defaults to instance value)
            max_entities: Maximum entities to include (defaults to instance value)
            
        Returns:
            Dictionary with pruning statistics
        """
        result = {
            'success': False,
            'entities_before': 0,
            'entities_after': 0,
            'relationships_before': 0,
            'relationships_after': 0,
            'processing_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Set parameters
            min_entity_freq = min_entity_freq or self.min_entity_freq
            min_relationship_weight = min_relationship_weight or self.min_relationship_weight
            max_entities = max_entities or self.max_entities
            
            # Count before pruning
            entity_count_before = sum(1 for node, data in self.graph.nodes(data=True) if data.get('type') == 'entity')
            edge_count_before = self.graph.number_of_edges()
            
            result['entities_before'] = entity_count_before
            result['relationships_before'] = edge_count_before
            
            # Create a new graph for pruned content
            pruned_graph = nx.DiGraph()
            
            # Add document nodes
            for node, data in self.graph.nodes(data=True):
                if data.get('type') == 'document':
                    pruned_graph.add_node(node, **data)
            
            # Add top entities based on frequency
            top_entities = [
                entity for entity, count in self.entity_counts.most_common(max_entities)
                if count >= min_entity_freq
            ]
            
            entity_nodes = {}
            for node, data in self.graph.nodes(data=True):
                if data.get('type') == 'entity' and data.get('text') in top_entities:
                    pruned_graph.add_node(node, **data)
                    entity_nodes[node] = data
            
            # Add edges between remaining nodes with sufficient weight
            for u, v, data in self.graph.edges(data=True):
                if (pruned_graph.has_node(u) and pruned_graph.has_node(v) and
                    data.get('weight', 0) >= min_relationship_weight):
                    pruned_graph.add_edge(u, v, **data)
            
            # Replace the graph with the pruned version
            self.graph = pruned_graph
            
            # Count after pruning
            entity_count_after = sum(1 for node, data in self.graph.nodes(data=True) if data.get('type') == 'entity')
            edge_count_after = self.graph.number_of_edges()
            
            result['entities_after'] = entity_count_after
            result['relationships_after'] = edge_count_after
            result['success'] = True
            
        except Exception as e:
            logger.error(f"Error pruning knowledge graph: {e}")
            result['error'] = str(e)
        
        # Calculate processing time
        result['processing_time'] = time.time() - start_time
        
        return result
    
    def get_entity_neighbors(self, entity_text: str, max_distance: int = 2) -> Dict[str, Any]:
        """
        Get neighboring entities for a specific entity.
        
        This method implements a graph traversal to find connected entities
        within a specified distance from the target entity.
        
        Args:
            entity_text: Text of the entity to explore
            max_distance: Maximum distance (hops) to traverse
            
        Returns:
            Dictionary with entity neighborhood information
        """
        result = {
            'success': False,
            'entity': entity_text,
            'neighbors': [],
            'error': None
        }
        
        try:
            # Find matching entity nodes
            matching_nodes = []
            for node, data in self.graph.nodes(data=True):
                if data.get('type') == 'entity' and data.get('text') == entity_text:
                    matching_nodes.append(node)
            
            if not matching_nodes:
                result['error'] = f"Entity '{entity_text}' not found in the knowledge graph."
                return result
            
            # Get neighbors for each matching node
            neighbors = set()
            explored = set(matching_nodes)
            
            # Breadth-first search to find neighbors
            current_frontier = set(matching_nodes)
            
            for distance in range(1, max_distance + 1):
                next_frontier = set()
                
                for node in current_frontier:
                    # Get incoming and outgoing neighbors
                    for neighbor in set(self.graph.successors(node)).union(self.graph.predecessors(node)):
                        if neighbor not in explored:
                            neighbor_data = self.graph.nodes[neighbor]
                            
                            if neighbor_data.get('type') == 'entity':
                                # Add to neighbors with distance information
                                neighbors.add((
                                    neighbor,
                                    neighbor_data.get('text', ''),
                                    neighbor_data.get('label', ''),
                                    distance
                                ))
                            
                            next_frontier.add(neighbor)
                            explored.add(neighbor)
                
                current_frontier = next_frontier
            
            # Format neighbors
            formatted_neighbors = []
            for node_id, text, label, distance in neighbors:
                # Get connecting relationships
                relationships = []
                
                for matching_node in matching_nodes:
                    # Check direct relationships
                    if self.graph.has_edge(matching_node, node_id):
                        data = self.graph[matching_node][node_id]
                        relationships.append({
                            'type': data.get('relationship', 'related'),
                            'weight': data.get('weight', 1),
                            'direction': 'outgoing'
                        })
                    
                    if self.graph.has_edge(node_id, matching_node):
                        data = self.graph[node_id][matching_node]
                        relationships.append({
                            'type': data.get('relationship', 'related'),
                            'weight': data.get('weight', 1),
                            'direction': 'incoming'
                        })
                
                formatted_neighbors.append({
                    'node_id': node_id,
                    'text': text,
                    'type': label,
                    'distance': distance,
                    'relationships': relationships
                })
            
            # Sort by distance then by relationship weight
            result['neighbors'] = sorted(
                formatted_neighbors,
                key=lambda x: (x['distance'], -max(rel['weight'] for rel in x['relationships']) if x['relationships'] else 0)
            )
            
            result['success'] = True
            
        except Exception as e:
            logger.error(f"Error getting entity neighbors: {e}")
            result['error'] = str(e)
        
        return result
    
    def find_paths(self, source_entity: str, target_entity: str, max_paths: int = 3) -> Dict[str, Any]:
        """
        Find paths between two entities in the knowledge graph.
        
        This method implements path-finding algorithms to discover connections
        between entities, revealing how concepts are related. It employs a k-shortest
        paths algorithm to identify multiple meaningful connections.
        
        Args:
            source_entity: Source entity text
            target_entity: Target entity text
            max_paths: Maximum number of paths to return
            
        Returns:
            Dictionary with path information including path details and metrics
        """
        result = {
            'success': False,
            'source': source_entity,
            'target': target_entity,
            'paths': [],
            'error': None
        }
        
        try:
            # Find matching source nodes
            source_nodes = []
            for node, data in self.graph.nodes(data=True):
                if data.get('type') == 'entity' and data.get('text') == source_entity:
                    source_nodes.append(node)
            
            if not source_nodes:
                result['error'] = f"Source entity '{source_entity}' not found in the knowledge graph."
                return result
            
            # Find matching target nodes
            target_nodes = []
            for node, data in self.graph.nodes(data=True):
                if data.get('type') == 'entity' and data.get('text') == target_entity:
                    target_nodes.append(node)
            
            if not target_nodes:
                result['error'] = f"Target entity '{target_entity}' not found in the knowledge graph."
                return result
            
            # Find paths between each source-target pair
            all_paths = []
            
            # Use k-shortest paths algorithm for each source-target pair
            for source_node in source_nodes:
                for target_node in target_nodes:
                    try:
                        # Find simple paths between source and target
                        simple_paths = list(nx.shortest_simple_paths(self.graph, source_node, target_node))
                        
                        # Limit to max_paths
                        paths = simple_paths[:max_paths]
                        
                        for path in paths:
                            path_info = self._format_path(path)
                            all_paths.append(path_info)
                    except nx.NetworkXNoPath:
                        # No path exists between this source-target pair
                        continue
            
            # Sort paths by length, then by total relationship weight
            all_paths.sort(key=lambda p: (len(p['nodes']), -p['total_weight']))
            
            # Limit to max_paths across all source-target pairs
            result['paths'] = all_paths[:max_paths]
            result['total_paths_found'] = len(all_paths)
            result['success'] = True
            
        except Exception as e:
            logger.error(f"Error finding paths between entities: {e}")
            result['error'] = str(e)
        
        return result
    
    def _format_path(self, path: List[str]) -> Dict[str, Any]:
        """
        Format a path between entities with detailed information.
        
        This helper method transforms a raw path into a structured representation
        with node and edge details for better interpretability.
        
        Args:
            path: List of node IDs representing a path
            
        Returns:
            Dictionary with formatted path information
        """
        nodes = []
        edges = []
        total_weight = 0
        
        # Add nodes with details
        for node_id in path:
            node_data = self.graph.nodes[node_id]
            node_info = {
                'id': node_id,
                'type': node_data.get('type', 'unknown'),
                'text': node_data.get('text', node_id) if node_data.get('type') == 'entity' else node_id
            }
            
            # Add entity-specific information
            if node_data.get('type') == 'entity':
                node_info['label'] = node_data.get('label', '')
                node_info['count'] = node_data.get('count', 0)
            
            # Add document-specific information
            if node_data.get('type') == 'document':
                node_info['metadata'] = node_data.get('metadata', {})
            
            nodes.append(node_info)
        
        # Add edges with details
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i+1]
            
            edge_data = self.graph.get_edge_data(source, target)
            weight = edge_data.get('weight', 1)
            relationship = edge_data.get('relationship', 'related')
            
            edge_info = {
                'source': source,
                'target': target,
                'relationship': relationship,
                'weight': weight
            }
            
            edges.append(edge_info)
            total_weight += weight
        
        # Create the path information
        path_info = {
            'nodes': nodes,
            'edges': edges,
            'length': len(path),
            'total_weight': total_weight,
            'average_weight': total_weight / len(edges) if edges else 0
        }
        
        return path_info
    
    def _extract_entity_relationships(self, doc, entities: List[Dict[str, Any]]) -> List[Tuple]:
        """
        Extract relationships between entities within a document.
        
        This method employs sophisticated NLP analysis to detect and categorize
        relationships between entities based on syntactic and semantic patterns.
        
        Args:
            doc: spaCy document object
            entities: List of extracted entities
            
        Returns:
            List of tuples containing (source_entity, target_entity, relationship_type)
        """
        relationships = []
        
        # Create a mapping of character positions to entities
        entity_map = {}
        for entity in entities:
            for pos in range(entity['start'], entity['end']):
                entity_map[pos] = entity
        
        # Create a set of spans for the entities
        entity_spans = {(e['start'], e['end']): e for e in entities}
        
        # Check for verb-mediated relationships
        for token in doc:
            if token.pos_ == "VERB":
                # Find subject and object connected to this verb
                subject = None
                direct_object = None
                
                # Extract subject
                for child in token.children:
                    if child.dep_ in ('nsubj', 'nsubjpass'):
                        # Get the root of the subject phrase
                        subject_root = child
                        while subject_root.head.dep_ == 'conj' and subject_root.head.pos_ != 'VERB':
                            subject_root = subject_root.head
                        
                        # Find if this token is part of a named entity
                        for i in range(len(subject_root.text)):
                            pos = subject_root.idx + i
                            if pos in entity_map:
                                subject = entity_map[pos]
                                break
                
                # Extract object
                for child in token.children:
                    if child.dep_ in ('dobj', 'pobj', 'attr'):
                        # Get the root of the object phrase
                        object_root = child
                        while object_root.head.dep_ == 'conj' and object_root.head.pos_ != 'VERB':
                            object_root = object_root.head
                        
                        # Find if this token is part of a named entity
                        for i in range(len(object_root.text)):
                            pos = object_root.idx + i
                            if pos in entity_map:
                                direct_object = entity_map[pos]
                                break
                
                # If we have both subject and object entities, create a relationship
                if subject and direct_object and subject != direct_object:
                    # Use the verb as relationship
                    relationship = token.lemma_.lower()
                    relationships.append((subject, direct_object, relationship))
        
        # Check for prepositional relationships
        for token in doc:
            if token.dep_ == 'pobj' and token.head.dep_ == 'prep':
                prep = token.head
                verb_or_noun = prep.head
                
                # Find the entities involved
                prep_object = None
                subject = None
                
                # Find the prepositional object entity
                for i in range(len(token.text)):
                    pos = token.idx + i
                    if pos in entity_map:
                        prep_object = entity_map[pos]
                        break
                
                # Find the subject entity
                if verb_or_noun.pos_ in ('NOUN', 'PROPN'):
                    for i in range(len(verb_or_noun.text)):
                        pos = verb_or_noun.idx + i
                        if pos in entity_map:
                            subject = entity_map[pos]
                            break
                
                # If we have both entities, create a relationship
                if subject and prep_object and subject != prep_object:
                    # Use the preposition as relationship (e.g., "in", "at", "of")
                    relationship = prep.text.lower()
                    relationships.append((subject, prep_object, relationship))
        
        # Check for direct entity-entity relationships based on proximity
        # This captures relationships that may not be explicitly stated through syntax
        sorted_entities = sorted(entities, key=lambda e: e['start'])
        
        for i in range(len(sorted_entities) - 1):
            current = sorted_entities[i]
            next_entity = sorted_entities[i + 1]
            
            # Check if entities are reasonably close (within 10 tokens)
            if next_entity['start'] - current['end'] < 50:
                # Find the tokens between these entities
                between_tokens = []
                for token in doc:
                    if current['end'] <= token.idx < next_entity['start']:
                        between_tokens.append(token)
                
                # Use between tokens to determine relationship type
                relationship = "related"
                
                # Check for specific relationship patterns in between tokens
                for token in between_tokens:
                    if token.pos_ == "ADP":  # Preposition
                        relationship = token.text.lower()
                        break
                    elif token.pos_ == "VERB":
                        relationship = token.lemma_.lower()
                        break
                
                relationships.append((current, next_entity, relationship))
        
        return relationships
    
    def export_graph(self, format: str = "json", path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export the knowledge graph in various formats.
        
        This method supports exporting the graph to multiple formats for
        visualization and analysis in external tools.
        
        Args:
            format: Export format ('json', 'graphml', 'gexf', 'cytoscape')
            path: Optional file path to save the export
            
        Returns:
            Dictionary with export results and data
        """
        result = {
            'success': False,
            'format': format,
            'node_count': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges(),
            'path': path,
            'data': None,
            'error': None
        }
        
        try:
            if format == "json":
                # Export as JSON
                data = {
                    'nodes': [],
                    'edges': []
                }
                
                # Add nodes
                for node, node_data in self.graph.nodes(data=True):
                    node_type = node_data.get('type', 'unknown')
                    node_info = {
                        'id': node,
                        'type': node_type
                    }
                    
                    # Add entity-specific properties
                    if node_type == 'entity':
                        node_info.update({
                            'text': node_data.get('text', ''),
                            'label': node_data.get('label', ''),
                            'count': node_data.get('count', 0)
                        })
                    
                    # Add document-specific properties
                    if node_type == 'document':
                        node_info.update({
                            'metadata': node_data.get('metadata', {})
                        })
                    
                    data['nodes'].append(node_info)
                
                # Add edges
                for source, target, edge_data in self.graph.edges(data=True):
                    edge_info = {
                        'source': source,
                        'target': target,
                        'relationship': edge_data.get('relationship', 'related'),
                        'weight': edge_data.get('weight', 1)
                    }
                    data['edges'].append(edge_info)
                
                # Save to file if path provided
                if path:
                    with open(path, 'w') as f:
                        json.dump(data, f, indent=2)
                
                result['data'] = data
                
            elif format in ["graphml", "gexf"]:
                # Export using NetworkX built-in functions
                if path:
                    if format == "graphml":
                        nx.write_graphml(self.graph, path)
                    else:  # gexf
                        nx.write_gexf(self.graph, path)
                    
                    result['data'] = f"Graph exported to {path} in {format} format"
                else:
                    result['error'] = f"Path must be provided for {format} export"
                    return result
                
            elif format == "cytoscape":
                # Export in Cytoscape.js format
                data = {
                    'nodes': [],
                    'edges': []
                }
                
                # Add nodes with Cytoscape.js properties
                for node, node_data in self.graph.nodes(data=True):
                    node_type = node_data.get('type', 'unknown')
                    node_info = {
                        'data': {
                            'id': node,
                            'type': node_type
                        }
                    }
                    
                    # Add entity-specific properties
                    if node_type == 'entity':
                        node_info['data'].update({
                            'name': node_data.get('text', ''),
                            'entity_type': node_data.get('label', ''),
                            'count': node_data.get('count', 0)
                        })
                    
                    # Add document-specific properties
                    if node_type == 'document':
                        node_info['data'].update({
                            'name': node_data.get('metadata', {}).get('source', node)
                        })
                    
                    data['nodes'].append(node_info)
                
                # Add edges with Cytoscape.js properties
                for source, target, edge_data in self.graph.edges(data=True):
                    edge_info = {
                        'data': {
                            'id': f"{source}-{target}",
                            'source': source,
                            'target': target,
                            'relationship': edge_data.get('relationship', 'related'),
                            'weight': edge_data.get('weight', 1)
                        }
                    }
                    data['edges'].append(edge_info)
                
                # Save to file if path provided
                if path:
                    with open(path, 'w') as f:
                        json.dump(data, f, indent=2)
                
                result['data'] = data
                
            else:
                result['error'] = f"Unsupported export format: {format}"
                return result
            
            result['success'] = True
            
        except Exception as e:
            logger.error(f"Error exporting knowledge graph: {e}")
            result['error'] = str(e)
        
        return result
    
    def find_central_entities(self, top_n: int = 10, algorithm: str = "pagerank") -> Dict[str, Any]:
        """
        Find the most central entities in the knowledge graph.
        
        This method applies advanced graph theory algorithms to identify
        the most influential or central entities in the knowledge graph.
        
        Args:
            top_n: Number of top entities to return
            algorithm: Centrality algorithm to use ('pagerank', 'betweenness', 'degree')
            
        Returns:
            Dictionary with centrality analysis results
        """
        result = {
            'success': False,
            'algorithm': algorithm,
            'top_entities': [],
            'error': None
        }
        
        try:
            # Create subgraph of only entity nodes
            entity_nodes = [
                node for node, data in self.graph.nodes(data=True)
                if data.get('type') == 'entity'
            ]
            entity_graph = self.graph.subgraph(entity_nodes)
            
            # Compute centrality based on selected algorithm
            if algorithm == "pagerank":
                centrality = nx.pagerank(entity_graph, weight='weight')
            elif algorithm == "betweenness":
                centrality = nx.betweenness_centrality(entity_graph, weight='weight')
            elif algorithm == "degree":
                centrality = nx.degree_centrality(entity_graph)
            else:
                result['error'] = f"Unsupported centrality algorithm: {algorithm}"
                return result
            
            # Get top entities
            top_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            # Format results
            formatted_entities = []
            for node_id, centrality_value in top_entities:
                node_data = self.graph.nodes[node_id]
                
                entity_info = {
                    'id': node_id,
                    'text': node_data.get('text', ''),
                    'label': node_data.get('label', ''),
                    'count': node_data.get('count', 0),
                    'centrality': centrality_value,
                    'connections': self.graph.degree(node_id)
                }
                
                formatted_entities.append(entity_info)
            
            result['top_entities'] = formatted_entities
            result['success'] = True
            
        except Exception as e:
            logger.error(f"Error finding central entities: {e}")
            result['error'] = str(e)
        
        return result
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the knowledge graph.
        
        This method calculates various graph metrics and statistics to provide
        insights into the structure and characteristics of the knowledge graph.
        
        Returns:
            Dictionary with detailed graph statistics
        """
        stats = {
            'node_count': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges(),
            'entity_count': sum(1 for _, data in self.graph.nodes(data=True) if data.get('type') == 'entity'),
            'document_count': sum(1 for _, data in self.graph.nodes(data=True) if data.get('type') == 'document'),
            'entity_types': list(self.entity_types),
            'relationship_types': list(self.relationship_types),
            'density': nx.density(self.graph),
            'is_directed': nx.is_directed(self.graph),
            'performance': {
                'avg_extraction_time': sum(self.extraction_times) / len(self.extraction_times) if self.extraction_times else 0,
                'total_entities_found': self.total_entities_found,
                'total_relationships_found': self.total_relationships_found
            }
        }
        
        # Calculate connected components
        if not nx.is_directed(self.graph):
            connected_components = list(nx.connected_components(self.graph))
            stats['connected_components'] = len(connected_components)
            stats['largest_component_size'] = len(max(connected_components, key=len)) if connected_components else 0
        else:
            strongly_connected = list(nx.strongly_connected_components(self.graph))
            weakly_connected = list(nx.weakly_connected_components(self.graph))
            stats['strongly_connected_components'] = len(strongly_connected)
            stats['weakly_connected_components'] = len(weakly_connected)
            stats['largest_strongly_connected_component'] = len(max(strongly_connected, key=len)) if strongly_connected else 0
        
        # Top entity types
        entity_type_counts = Counter()
        for _, data in self.graph.nodes(data=True):
            if data.get('type') == 'entity':
                entity_type_counts[data.get('label', 'unknown')] += 1
        
        stats['entity_type_distribution'] = dict(entity_type_counts.most_common())
        
        # Top relationship types
        relationship_counts = Counter()
        for _, _, data in self.graph.edges(data=True):
            relationship_counts[data.get('relationship', 'unknown')] += 1
        
        stats['relationship_type_distribution'] = dict(relationship_counts.most_common())
        
        return stats
