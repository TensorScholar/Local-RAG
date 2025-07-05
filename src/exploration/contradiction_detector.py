"""
Contradiction Detection Module.

This module implements sophisticated algorithms for identifying contradictions
and inconsistencies across documents in the knowledge base. It focuses on
detecting numerical inconsistencies, date contradictions, entity conflicts,
and factual contradictions through advanced NLP and logical analysis.

Implementation features:
- Numerical range analysis with statistical significance testing
- Temporal reasoning with date normalization and conflict detection
- Entity attribute comparison with inconsistency identification
- Cross-document claim validation with confidence scoring
"""

import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from collections import defaultdict
import spacy
import numpy as np
from datetime import datetime
import dateutil.parser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContradictionDetector:
    """
    Detect contradictions and inconsistencies across documents.
    
    This class implements various contradiction detection algorithms to
    identify inconsistencies in information across the document collection.
    """
    
    def __init__(self, language_model: str = "en_core_web_sm"):
        """
        Initialize the contradiction detector with NLP capabilities.
        
        Args:
            language_model: spaCy language model to use for NLP
        """
        # Initialize NLP model
        try:
            self.nlp = spacy.load(language_model)
            logger.info(f"Loaded spaCy model: {language_model}")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            logger.error("Run: python -m spacy download en_core_web_sm")
            raise
        
        # Initialize contradiction types and patterns
        self.contradiction_types = {
            'numerical': self._detect_numerical_contradictions,
            'date': self._detect_date_contradictions,
            'entity': self._detect_entity_contradictions,
            'claim': self._detect_claim_contradictions
        }
        
        # Patterns for extracting various types of information
        self.number_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(%|percent|kg|kilometers|km|miles|mi|years|year|months|month|days|day|hours|hour)')
        self.date_pattern = re.compile(r'\b(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{2,4}|\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,? \d{2,4})\b')
        
        # Performance tracking
        self.total_documents_processed = 0
        self.total_contradictions_found = 0
        self.processing_times = []
    
    def detect_contradictions(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect contradictions across a set of documents.
        
        This method coordinates the detection of different types of contradictions
        across the document collection, aggregating results into a comprehensive report.
        
        Args:
            documents: List of document dictionaries with text and metadata
            
        Returns:
            Dictionary with contradiction detection results
        """
        result = {
            'success': False,
            'documents_processed': 0,
            'contradiction_count': 0,
            'contradictions': [],
            'processing_time': 0,
            'error': None
        }
        
        start_time = time.time()
        
        try:
            # Validate input
            if not documents:
                result['error'] = "No documents provided for contradiction detection."
                return result
            
            # Extract text and metadata
            processed_docs = []
            for doc in documents:
                if isinstance(doc, dict) and 'text' in doc:
                    doc_text = doc['text']
                    doc_metadata = doc.get('metadata', {})
                    doc_id = doc_metadata.get('id', None) or doc_metadata.get('source', f"doc_{len(processed_docs)}")
                    
                    # Process with spaCy for NLP features
                    nlp_doc = self.nlp(doc_text)
                    
                    processed_docs.append({
                        'id': doc_id,
                        'text': doc_text,
                        'metadata': doc_metadata,
                        'nlp_doc': nlp_doc
                    })
            
            # Detect contradictions of each type
            all_contradictions = []
            
            for contradiction_type, detection_function in self.contradiction_types.items():
                type_contradictions = detection_function(processed_docs)
                all_contradictions.extend(type_contradictions)
            
            # Sort contradictions by confidence score
            all_contradictions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Update results
            result['documents_processed'] = len(processed_docs)
            result['contradiction_count'] = len(all_contradictions)
            result['contradictions'] = all_contradictions
            
            # Update statistics
            self.total_documents_processed += len(processed_docs)
            self.total_contradictions_found += len(all_contradictions)
            
            result['success'] = True
            
        except Exception as e:
            logger.error(f"Error detecting contradictions: {e}")
            result['error'] = str(e)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        result['processing_time'] = processing_time
        
        return result
    
    def _detect_numerical_contradictions(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect contradictions in numerical values across documents.
        
        This method identifies inconsistencies in numerical data, such as
        different figures reported for the same metric across documents.
        
        Args:
            documents: List of processed document dictionaries
            
        Returns:
            List of numerical contradiction dictionaries
        """
        contradictions = []
        
        # Extract all numerical mentions with context
        numerical_mentions = []
        
        for doc in documents:
            doc_id = doc['id']
            text = doc['text']
            
            # Find numerical values with units
            for match in self.number_pattern.finditer(text):
                value_str, unit = match.groups()
                try:
                    value = float(value_str)
                    
                    # Get context (surrounding text)
                    start_idx = max(0, match.start() - 100)
                    end_idx = min(len(text), match.end() + 100)
                    context = text[start_idx:end_idx]
                    
                    # Analyze context to determine what the number refers to
                    context_doc = self.nlp(context)
                    subject = self._extract_subject_from_context(context_doc, match.start() - start_idx)
                    
                    if subject:
                        numerical_mentions.append({
                            'document_id': doc_id,
                            'value': value,
                            'unit': unit,
                            'subject': subject,
                            'context': context,
                            'position': match.start()
                        })
                except ValueError:
                    continue
        
        # Group numerical mentions by subject and unit
        grouped_mentions = defaultdict(list)
        for mention in numerical_mentions:
            key = (mention['subject'].lower(), mention['unit'])
            grouped_mentions[key].append(mention)
        
        # Check for contradictions in each group
        for (subject, unit), mentions in grouped_mentions.items():
            if len(mentions) > 1:
                # Check if there are significant differences in the values
                values = [mention['value'] for mention in mentions]
                mean_value = np.mean(values)
                std_dev = np.std(values)
                
                # Determine if there's significant variation
                # High variation coefficient indicates potential contradiction
                if len(values) >= 2 and std_dev / mean_value > 0.2:  # 20% threshold
                    # Find the mentions with the most divergent values
                    min_value_mention = min(mentions, key=lambda m: m['value'])
                    max_value_mention = max(mentions, key=lambda m: m['value'])
                    
                    # Create contradiction
                    contradiction = {
                        'type': 'numerical',
                        'subject': subject,
                        'unit': unit,
                        'values': values,
                        'mean': mean_value,
                        'std_dev': std_dev,
                        'variation_coefficient': std_dev / mean_value,
                        'conflicting_documents': [
                            {
                                'document_id': min_value_mention['document_id'],
                                'value': min_value_mention['value'],
                                'context': min_value_mention['context']
                            },
                            {
                                'document_id': max_value_mention['document_id'],
                                'value': max_value_mention['value'],
                                'context': max_value_mention['context']
                            }
                        ],
                        'confidence': min(1.0, (std_dev / mean_value) * 2),  # Scale confidence by variation
                        'description': f"Conflicting numerical values for '{subject}' ({unit}): {min_value_mention['value']} vs {max_value_mention['value']}"
                    }
                    
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _detect_date_contradictions(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect contradictions in dates across documents.
        
        This method identifies inconsistencies in dates, such as different
        dates reported for the same event across documents.
        
        Args:
            documents: List of processed document dictionaries
            
        Returns:
            List of date contradiction dictionaries
        """
        contradictions = []
        
        # Extract all date mentions with context
        date_mentions = []
        
        for doc in documents:
            doc_id = doc['id']
            text = doc['text']
            
            # Find date values
            for match in self.date_pattern.finditer(text):
                date_str = match.group(0)
                try:
                    # Parse the date
                    parsed_date = dateutil.parser.parse(date_str)
                    
                    # Get context (surrounding text)
                    start_idx = max(0, match.start() - 100)
                    end_idx = min(len(text), match.end() + 100)
                    context = text[start_idx:end_idx]
                    
                    # Analyze context to determine what the date refers to
                    context_doc = self.nlp(context)
                    event = self._extract_event_from_context(context_doc, match.start() - start_idx)
                    
                    if event:
                        date_mentions.append({
                            'document_id': doc_id,
                            'date': parsed_date,
                            'date_str': date_str,
                            'event': event,
                            'context': context,
                            'position': match.start()
                        })
                except (ValueError, dateutil.parser.ParserError):
                    continue
        
        # Group date mentions by event
        grouped_mentions = defaultdict(list)
        for mention in date_mentions:
            key = mention['event'].lower()
            grouped_mentions[key].append(mention)
        
        # Check for contradictions in each group
        for event, mentions in grouped_mentions.items():
            if len(mentions) > 1:
                # Sort mentions by date
                sorted_mentions = sorted(mentions, key=lambda m: m['date'])
                
                # Check if dates differ by more than 7 days
                first_mention = sorted_mentions[0]
                last_mention = sorted_mentions[-1]
                date_diff = (last_mention['date'] - first_mention['date']).days
                
                if abs(date_diff) > 7:  # More than a week difference
                    # Create contradiction
                    contradiction = {
                        'type': 'date',
                        'event': event,
                        'date_difference_days': date_diff,
                        'conflicting_documents': [
                            {
                                'document_id': first_mention['document_id'],
                                'date': first_mention['date_str'],
                                'context': first_mention['context']
                            },
                            {
                                'document_id': last_mention['document_id'],
                                'date': last_mention['date_str'],
                                'context': last_mention['context']
                            }
                        ],
                        'confidence': min(1.0, abs(date_diff) / 365),  # Scale confidence by difference
                        'description': f"Conflicting dates for '{event}': {first_mention['date_str']} vs {last_mention['date_str']} ({date_diff} days apart)"
                    }
                    
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _detect_entity_contradictions(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect contradictions in entity attributes across documents.
        
        This method identifies inconsistencies in attributes assigned to the same entity,
        such as different titles, roles, or characteristics.
        
        Args:
            documents: List of processed document dictionaries
            
        Returns:
            List of entity contradiction dictionaries
        """
        contradictions = []
        
        # Extract entity mentions with attributes
        entity_attributes = defaultdict(lambda: defaultdict(list))
        
        for doc in documents:
            doc_id = doc['id']
            nlp_doc = doc['nlp_doc']
            
            # Process entities in the document
            for ent in nlp_doc.ents:
                if ent.label_ in ('PERSON', 'ORG', 'GPE', 'PRODUCT'):
                    # Look for attributes associated with this entity
                    for sent in nlp_doc.sents:
                        if ent.start >= sent.start and ent.end <= sent.end:
                            # Entity is in this sentence
                            # Extract attributes using dependency parsing
                            attributes = self._extract_entity_attributes(sent, ent)
                            
                            # Add attributes to the entity
                            for attr_type, attr_value in attributes:
                                entity_attributes[ent.text][attr_type].append({
                                    'value': attr_value,
                                    'document_id': doc_id,
                                    'context': sent.text
                                })
        
        # Check for contradictory attributes
        for entity, attributes in entity_attributes.items():
            for attr_type, attr_mentions in attributes.items():
                if len(attr_mentions) > 1:
                    # Group attribute values
                    value_groups = defaultdict(list)
                    for mention in attr_mentions:
                        value_groups[mention['value']].append(mention)
                    
                    # Check if there are multiple distinct values
                    if len(value_groups) > 1:
                        # Get the two most frequent conflicting values
                        top_values = sorted(value_groups.items(), key=lambda x: len(x[1]), reverse=True)[:2]
                        value1, mentions1 = top_values[0]
                        value2, mentions2 = top_values[1]
                        
                        # Create contradiction
                        contradiction = {
                            'type': 'entity',
                            'entity': entity,
                            'attribute': attr_type,
                            'conflicting_values': [value1, value2],
                            'conflicting_documents': [
                                {
                                    'document_id': mentions1[0]['document_id'],
                                    'value': value1,
                                    'context': mentions1[0]['context']
                                },
                                {
                                    'document_id': mentions2[0]['document_id'],
                                    'value': value2,
                                    'context': mentions2[0]['context']
                                }
                            ],
                            'confidence': 0.8,  # Default confidence for entity contradictions
                            'description': f"Conflicting attributes for '{entity}': {attr_type} is '{value1}' vs '{value2}'"
                        }
                        
                        contradictions.append(contradiction)
        
        return contradictions
    
    def _detect_claim_contradictions(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect contradictions in factual claims across documents.
        
        This method identifies inconsistencies in statements or claims made across
        different documents, focusing on contradictory assertions.
        
        Args:
            documents: List of processed document dictionaries
            
        Returns:
            List of claim contradiction dictionaries
        """
        contradictions = []
        
        # Extract claims from documents
        claims = []
        
        for doc in documents:
            doc_id = doc['id']
            nlp_doc = doc['nlp_doc']
            
            # Process sentences to identify claims
            for sent in nlp_doc.sents:
                # Focus on sentences that make definitive statements
                if self._is_claim_sentence(sent):
                    # Normalize the claim
                    normalized_claim = self._normalize_claim(sent)
                    
                    claims.append({
                        'document_id': doc_id,
                        'text': sent.text,
                        'normalized': normalized_claim,
                        'subject': self._extract_claim_subject(sent),
                        'polarity': self._detect_claim_polarity(sent)
                    })
        
        # Group claims by subject
        grouped_claims = defaultdict(list)
        for claim in claims:
            if claim['subject']:
                grouped_claims[claim['subject']].append(claim)
        
        # Check for contradictory claims
        for subject, subject_claims in grouped_claims.items():
            # Group by polarity
            positive_claims = [c for c in subject_claims if c['polarity'] > 0]
            negative_claims = [c for c in subject_claims if c['polarity'] < 0]
            
            # Check for contradictions between positive and negative claims
            if positive_claims and negative_claims:
                # Sort by polarity strength
                strongest_positive = max(positive_claims, key=lambda c: c['polarity'])
                strongest_negative = min(negative_claims, key=lambda c: c['polarity'])
                
                # Create contradiction
                contradiction = {
                    'type': 'claim',
                    'subject': subject,
                    'conflicting_documents': [
                        {
                            'document_id': strongest_positive['document_id'],
                            'claim': strongest_positive['text'],
                            'polarity': strongest_positive['polarity']
                        },
                        {
                            'document_id': strongest_negative['document_id'],
                            'claim': strongest_negative['text'],
                            'polarity': strongest_negative['polarity']
                        }
                    ],
                    'confidence': 0.7,  # Default confidence for claim contradictions
                    'description': f"Contradictory claims about '{subject}': {strongest_positive['text']} vs {strongest_negative['text']}"
                }
                
                contradictions.append(contradiction)
        
        return contradictions
    
    def _extract_subject_from_context(self, context_doc, position: int) -> Optional[str]:
        """
        Extract the subject that a numerical value refers to from context.
        
        Args:
            context_doc: spaCy Doc object of the context
            position: Position of the numerical value in the context
            
        Returns:
            Subject string or None if no subject found
        """
        # Find the token at or before the position
        target_token = None
        for token in context_doc:
            if token.idx <= position < token.idx + len(token.text):
                target_token = token
                break
        
        if not target_token:
            return None
        
        # Look for nouns preceding the number
        for noun in reversed(list(context_doc[:target_token.i])):
            if noun.pos_ in ('NOUN', 'PROPN') and noun.i > target_token.i - 10:
                # Check if it's part of a noun phrase
                for np in context_doc.noun_chunks:
                    if noun.i >= np.start and noun.i < np.end:
                        return np.text
                return noun.text
        
        # Look for nouns following the number
        for token in context_doc[target_token.i:]:
            if token.pos_ in ('NOUN', 'PROPN') and token.i < target_token.i + 5:
                # Check if it's part of a noun phrase
                for np in context_doc.noun_chunks:
                    if token.i >= np.start and token.i < np.end:
                        return np.text
                return token.text
        
        return None
    
    def _extract_event_from_context(self, context_doc, position: int) -> Optional[str]:
        """
        Extract the event that a date refers to from context.
        
        Args:
            context_doc: spaCy Doc object of the context
            position: Position of the date in the context
            
        Returns:
            Event string or None if no event found
        """
        # Find the token at or before the position
        target_token = None
        for token in context_doc:
            if token.idx <= position < token.idx + len(token.text):
                target_token = token
                break
        
        if not target_token:
            return None
        
        # Look for event-related verbs and nouns
        event_tokens = []
        event_verbs = ('occur', 'happen', 'begin', 'start', 'end', 'launch', 'announce', 'release', 'publish', 'establish', 'found', 'create')
        
        # Check for event-related prepositions
        prep_position = False
        for i in range(max(0, target_token.i - 3), min(len(context_doc), target_token.i + 3)):
            if context_doc[i].text.lower() in ('on', 'in', 'at') and i < target_token.i:
                prep_position = True
                break
        
        # If date is preceded by a preposition, look for noun phrase after it
        if prep_position:
            for token in context_doc[target_token.i:]:
                if token.pos_ in ('NOUN', 'PROPN') and token.i < target_token.i + 10:
                    for np in context_doc.noun_chunks:
                        if token.i >= np.start and token.i < np.end:
                            return np.text
                    event_tokens.append(token.text)
                    break
        else:
            # Otherwise, look for event indicators near the date
            for i in range(max(0, target_token.i - 15), min(len(context_doc), target_token.i + 15)):
                token = context_doc[i]
                # Check for event-related verbs
                if token.lemma_ in event_verbs:
                    # Get the subject of this verb
                    for child in token.children:
                        if child.dep_ in ('nsubj', 'nsubjpass'):
                            for np in context_doc.noun_chunks:
                                if child.i >= np.start and child.i < np.end:
                                    return np.text
                            event_tokens.append(child.text)
                            break
                # Check for event nouns
                elif token.pos_ in ('NOUN', 'PROPN') and i != target_token.i:
                    if token.text.lower() in ('event', 'conference', 'meeting', 'release', 'launch', 'announcement', 'publication', 'election', 'ceremony', 'festival', 'celebration'):
                        for np in context_doc.noun_chunks:
                            if token.i >= np.start and token.i < np.end:
                                return np.text
                        event_tokens.append(token.text)
                        break
        
        # If we found event tokens, join them
        if event_tokens:
            return ' '.join(event_tokens)
        
        # Fallback: Use the nearest noun phrase
        min_distance = float('inf')
        closest_np = None
        
        for np in context_doc.noun_chunks:
            # Calculate distance to the target token
            np_position = (np.start_char + np.end_char) // 2
            distance = abs(np_position - position)
            
            if distance < min_distance:
                min_distance = distance
                closest_np = np
        
        return closest_np.text if closest_np and min_distance < 50 else None
    
    def _extract_entity_attributes(self, sent, entity) -> List[Tuple[str, str]]:
        """
        Extract attributes associated with an entity in a sentence.
        
        Args:
            sent: spaCy Span representing a sentence
            entity: spaCy Span representing a named entity
            
        Returns:
            List of (attribute_type, attribute_value) tuples
        """
        attributes = []
        
        # Find the entity's root token
        entity_tokens = [sent[i] for i in range(len(sent)) if sent[i].idx >= entity.start_char and sent[i].idx < entity.end_char]
        if not entity_tokens:
            return attributes
        
        entity_root = entity_tokens[-1]
        
        # Check if entity is subject of a copular verb ("X is Y")
        for token in sent:
            if token.dep_ == 'ROOT' and token.lemma_ == 'be':
                # Check if entity is the subject
                for child in token.children:
                    if child.dep_ == 'nsubj' and child.idx >= entity.start_char and child.idx < entity.end_char:
                        # Find the complement (attribute)
                        for comp in token.children:
                            if comp.dep_ in ('attr', 'acomp'):
                                # Get the full attribute phrase
                                attr_tokens = [comp]
                                for descendant in comp.subtree:
                                    if descendant != comp:
                                        attr_tokens.append(descendant)
                                
                                attr_tokens.sort(key=lambda t: t.i)
                                attr_value = ' '.join(t.text for t in attr_tokens)
                                attributes.append(('identity', attr_value))
        
        # Check for appositive phrases ("X, the Y")
        if entity_root.dep_ == 'appos' or any(child.dep_ == 'appos' for child in entity_root.children):
            if entity_root.dep_ == 'appos':
                head = entity_root.head
                if head.pos_ in ('NOUN', 'PROPN'):
                    # Entity is the appositive, head is the noun it describes
                    appos_tokens = [t for t in entity_root.subtree]
                    appos_tokens.sort(key=lambda t: t.i)
                    appos_value = ' '.join(t.text for t in appos_tokens)
                    attributes.append(('appositive', appos_value))
            else:
                # Entity has an appositive
                for child in entity_root.children:
                    if child.dep_ == 'appos':
                        appos_tokens = [t for t in child.subtree]
                        appos_tokens.sort(key=lambda t: t.i)
                        appos_value = ' '.join(t.text for t in appos_tokens)
                        attributes.append(('appositive', appos_value))
        
        # Check for age information
        age_pattern = re.compile(r'\b(\d+)(?:-year-old|years old|\syo)\b')
        age_match = age_pattern.search(sent.text)
        if age_match and entity.start_char < age_match.start() < entity.end_char + 30:
            attributes.append(('age', age_match.group(1)))
        
        # Check for titles or roles (preceding the entity)
        title_pos = entity.start_char
        title_tokens = []
        
        for token in reversed(list(sent)):
            if token.idx + len(token.text) <= entity.start_char and len(title_tokens) < 5:
                if token.pos_ in ('NOUN', 'PROPN', 'ADJ') or token.text.lower() in ('mr', 'mrs', 'ms', 'dr', 'prof'):
                    title_tokens.insert(0, token)
                else:
                    break
        
        if title_tokens:
            title = ' '.join(t.text for t in title_tokens)
            attributes.append(('title', title))
        
        return attributes
    
    def _is_claim_sentence(self, sent) -> bool:
        """
        Determine if a sentence makes a factual claim.
        
        Args:
            sent: spaCy Span representing a sentence
            
        Returns:
            True if the sentence makes a claim, False otherwise
        """
        # Check for definitive statements
        # Claim sentences typically have a subject-verb structure without hedging
        has_subject = any(token.dep_ in ('nsubj', 'nsubjpass') for token in sent)
        has_verb = any(token.pos_ == 'VERB' for token in sent)
        
        # Check for hedging language (reduces confidence in claims)
        hedging_terms = ('may', 'might', 'could', 'possibly', 'perhaps', 'likely', 'unlikely', 'potential', 'potential', 'allegedly')
        has_hedging = any(token.lemma_ in hedging_terms for token in sent)
        
        # Check for opinion markers
        opinion_markers = ('believe', 'think', 'feel', 'opinion', 'view', 'perspective', 'suggest')
        has_opinion = any(token.lemma_ in opinion_markers for token in sent)
        
        # Check sentence length (very short or very long sentences are less likely to be claims)
        good_length = 5 <= len(sent) <= 40
        
        return has_subject and has_verb and good_length and not has_hedging and not has_opinion
    
    def _normalize_claim(self, sent) -> str:
        """
        Normalize a claim sentence for better comparison.
        
        Args:
            sent: spaCy Span representing a sentence
            
        Returns:
            Normalized claim string
        """
        # Convert to lowercase
        text = sent.text.lower()
        
        # Remove punctuation except apostrophes
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stop words except negations
        important_stops = ('no', 'not', 'never', 'none', 'nothing', 'nor')
        tokens = [token.text for token in sent if not token.is_stop or token.text.lower() in important_stops]
        
        return ' '.join(tokens)
    
    def _extract_claim_subject(self, sent) -> Optional[str]:
        """
        Extract the subject of a claim.
        
        Args:
            sent: spaCy Span representing a sentence
            
        Returns:
            Subject string or None if no subject found
        """
        # Find the root verb
        root = None
        for token in sent:
            if token.dep_ == 'ROOT':
                root = token
                break
        
        if not root:
            return None
        
        # Find the subject of the root verb
        subject = None
        for token in root.children:
            if token.dep_ in ('nsubj', 'nsubjpass'):
                # Get the full subject phrase
                subject_tokens = [t for t in token.subtree if t.dep_ not in ('det', 'punct')]
                subject_tokens.sort(key=lambda t: t.i)
                subject = ' '.join(t.lemma_ for t in subject_tokens)
                break
        
        return subject
    
    def _detect_claim_polarity(self, sent) -> float:
        """
        Detect the polarity (positive/negative) of a claim.
        
        Args:
            sent: spaCy Span representing a sentence
            
        Returns:
            Polarity score (-1.0 to 1.0)
        """
        # Check for explicit negations
        negation_tokens = ('no', 'not', 'never', 'none', 'nothing', 'without', 'lack', 'absent', 'negative')
        negation_count = sum(1 for token in sent if token.lower_ in negation_tokens)
        
        # Even number of negations cancel out, odd number results in negation
        is_negated = negation_count % 2 == 1
        
        # Set base polarity
        polarity = -0.8 if is_negated else 0.8
        
        # Adjust based on intensity markers
        intensifiers = ('very', 'extremely', 'absolutely', 'definitely', 'certainly', 'completely')
        intensity_score = sum(0.1 for token in sent if token.lower_ in intensifiers)
        
        # Constrain final polarity to [-1.0, 1.0]
        final_polarity = max(-1.0, min(1.0, polarity + (intensity_score * (1 if polarity > 0 else -1))))
        
        return final_polarity
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for contradiction detection.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            'total_documents_processed': self.total_documents_processed,
            'total_contradictions_found': self.total_contradictions_found,
            'contradiction_types': {
                'numerical': 0,
                'date': 0,
                'entity': 0,
                'claim': 0
            },
            'average_processing_time': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        }
        
        return stats
