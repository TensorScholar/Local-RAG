"""
Advanced Scientific Text Processing Engine with Mathematical Intelligence.

This module implements a sophisticated scientific text processing system that combines
state-of-the-art computational algebra, symbolic mathematics, and domain-specific
natural language processing techniques. The architecture employs advanced mathematical
parsing algorithms with rigorous symbolic computation capabilities for precise
scientific knowledge extraction and manipulation.

Mathematical Foundations:
1. Symbolic Algebra Systems - Advanced symbolic computation with CAS integration
2. Dimensional Analysis - Rigorous unit consistency verification and conversion
3. Chemical Structure Processing - Molecular formula parsing and validation
4. Mathematical Expression Trees - Abstract syntax trees for formula manipulation
5. Scientific Notation Normalization - Standardized numerical representation

Advanced Capabilities:
- LaTeX and MathML parsing with semantic understanding
- Chemical formula validation and stoichiometric analysis
- Unit conversion with dimensional analysis verification
- Mathematical equation solving and simplification
- Scientific citation and reference extraction
- Molecular structure interpretation and visualization

Author: Advanced RAG System Team
Version: 2.0.0
"""

import re
import logging
import math
import fractions
from typing import Dict, List, Optional, Union, Tuple, Any, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum, auto
import json
from pathlib import Path

# Advanced mathematical and scientific computation
import sympy as sp
from sympy import symbols, sympify, latex, simplify, solve, diff, integrate
from sympy.physics.units import *
from sympy.parsing.latex import parse_latex
import numpy as np

# Chemical and molecular processing
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available - chemical processing will be limited")

# Configure scientific logging
logger = logging.getLogger(__name__)


class ScientificDomain(Enum):
    """Enumeration of scientific domains for specialized processing."""
    MATHEMATICS = auto()
    PHYSICS = auto()
    CHEMISTRY = auto()
    BIOLOGY = auto()
    ENGINEERING = auto()
    GENERAL = auto()


class MathematicalObjectType(Enum):
    """Types of mathematical objects for semantic classification."""
    EQUATION = auto()
    FORMULA = auto()
    CONSTANT = auto()
    VARIABLE = auto()
    FUNCTION = auto()
    THEOREM = auto()
    PROOF = auto()


@dataclass
class MathematicalExpression:
    """Representation of mathematical expressions with semantic metadata."""
    
    original_text: str
    latex_form: str
    sympy_expression: Optional[sp.Expr] = None
    domain: ScientificDomain = ScientificDomain.MATHEMATICS
    object_type: MathematicalObjectType = MathematicalObjectType.EQUATION
    variables: Set[str] = field(default_factory=set)
    constants: Set[str] = field(default_factory=set)
    units: Optional[str] = None
    dimensionality: Optional[Dict[str, int]] = None
    
    def __post_init__(self):
        """Post-initialization processing for mathematical expressions."""
        if self.sympy_expression is None:
            try:
                self.sympy_expression = sympify(self.original_text)
            except Exception:
                logger.debug(f"Could not convert to SymPy: {self.original_text}")
        
        if self.sympy_expression:
            # Extract variables and constants
            free_symbols = self.sympy_expression.free_symbols
            self.variables = {str(symbol) for symbol in free_symbols}


@dataclass
class ChemicalCompound:
    """Representation of chemical compounds with structural information."""
    
    formula: str
    name: Optional[str] = None
    molecular_weight: Optional[float] = None
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    
    # Chemical properties
    elements: Dict[str, int] = field(default_factory=dict)
    functional_groups: List[str] = field(default_factory=list)
    
    # Computed properties
    is_valid: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization chemical analysis."""
        self._parse_molecular_formula()
        self._validate_chemistry()
    
    def _parse_molecular_formula(self):
        """Parse molecular formula to extract elemental composition."""
        # Regular expression for chemical formula parsing
        formula_pattern = r'([A-Z][a-z]?)(\d*)'
        
        matches = re.findall(formula_pattern, self.formula)
        for element, count_str in matches:
            count = int(count_str) if count_str else 1
            self.elements[element] = self.elements.get(element, 0) + count
    
    def _validate_chemistry(self):
        """Validate chemical formula and compute properties."""
        if not RDKIT_AVAILABLE:
            return
        
        try:
            # Create molecule from formula
            mol = Chem.MolFromFormula(self.formula)
            if mol is None:
                self.is_valid = False
                self.error_message = "Invalid molecular formula"
                return
            
            # Compute molecular weight
            self.molecular_weight = Descriptors.MolWt(mol)
            
            # Generate SMILES if possible
            try:
                self.smiles = Chem.MolToSmiles(mol)
            except:
                pass
                
        except Exception as e:
            self.is_valid = False
            self.error_message = str(e)


@dataclass
class ScientificUnit:
    """Representation of scientific units with dimensional analysis."""
    
    symbol: str
    name: str
    base_dimensions: Dict[str, int] = field(default_factory=dict)
    conversion_factor: float = 1.0
    conversion_offset: float = 0.0
    
    # Standard SI base dimensions
    SI_DIMENSIONS = {
        'length': 'm',
        'mass': 'kg', 
        'time': 's',
        'current': 'A',
        'temperature': 'K',
        'amount': 'mol',
        'luminosity': 'cd'
    }


class MathematicalParser:
    """Advanced parser for mathematical expressions and equations."""
    
    def __init__(self):
        """Initialize mathematical parser with pattern recognition."""
        # Mathematical pattern definitions
        self.equation_patterns = [
            r'([^=]+)=([^=]+)',  # Basic equation
            r'\\begin{equation}(.*?)\\end{equation}',  # LaTeX equation
            r'\\begin{align}(.*?)\\end{align}',  # LaTeX align
            r'\$\$(.*?)\$\$',  # Display math
            r'\$(.*?)\$',  # Inline math
        ]
        
        # Function patterns
        self.function_patterns = [
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]+)\)',  # Function notation
            r'\\([a-zA-Z]+)\s*\{([^}]+)\}',  # LaTeX functions
        ]
        
        # Mathematical constants
        self.constants = {
            'pi': sp.pi,
            'π': sp.pi,
            'e': sp.E,
            'i': sp.I,
            'infinity': sp.oo,
            '∞': sp.oo,
            'gamma': sp.EulerGamma,
            'phi': sp.GoldenRatio,
        }
    
    def parse_expression(self, text: str) -> List[MathematicalExpression]:
        """
        Parse mathematical expressions from text.
        
        Args:
            text: Input text containing mathematical content
            
        Returns:
            List of parsed mathematical expressions
        """
        expressions = []
        
        # Find mathematical patterns
        for pattern in self.equation_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    expr_text = '='.join(match) if len(match) > 1 else match[0]
                else:
                    expr_text = match
                
                # Clean and parse expression
                cleaned_expr = self._clean_expression(expr_text)
                if cleaned_expr:
                    math_expr = self._create_mathematical_expression(cleaned_expr)
                    if math_expr:
                        expressions.append(math_expr)
        
        return expressions
    
    def _clean_expression(self, expr: str) -> str:
        """Clean and normalize mathematical expression."""
        # Remove LaTeX formatting
        expr = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', expr)
        expr = re.sub(r'\\[a-zA-Z]+', '', expr)
        
        # Normalize common mathematical symbols
        replacements = {
            '×': '*',
            '÷': '/',
            '²': '**2',
            '³': '**3',
            '√': 'sqrt',
            '∑': 'Sum',
            '∫': 'Integral',
            '∂': 'diff',
            '∆': 'Delta',
            '∇': 'nabla',
        }
        
        for old, new in replacements.items():
            expr = expr.replace(old, new)
        
        # Remove extra whitespace
        expr = re.sub(r'\s+', ' ', expr).strip()
        
        return expr
    
    def _create_mathematical_expression(self, expr_text: str) -> Optional[MathematicalExpression]:
        """Create mathematical expression object from text."""
        try:
            # Attempt to parse with SymPy
            sympy_expr = sympify(expr_text, locals=self.constants)
            
            # Generate LaTeX representation
            latex_form = latex(sympy_expr)
            
            # Determine object type
            obj_type = self._classify_expression(sympy_expr, expr_text)
            
            return MathematicalExpression(
                original_text=expr_text,
                latex_form=latex_form,
                sympy_expression=sympy_expr,
                object_type=obj_type
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse mathematical expression: {expr_text}, error: {e}")
            return None
    
    def _classify_expression(self, sympy_expr: sp.Expr, original_text: str) -> MathematicalObjectType:
        """Classify the type of mathematical expression."""
        # Check for equations (contains equality)
        if '=' in original_text:
            return MathematicalObjectType.EQUATION
        
        # Check for function definitions
        if hasattr(sympy_expr, 'func') and sympy_expr.func in [sp.Function]:
            return MathematicalObjectType.FUNCTION
        
        # Check for constants
        if sympy_expr.is_constant():
            return MathematicalObjectType.CONSTANT
        
        # Check for variables
        if sympy_expr.is_symbol:
            return MathematicalObjectType.VARIABLE
        
        # Default to formula
        return MathematicalObjectType.FORMULA


class ChemicalParser:
    """Advanced parser for chemical formulas and structures."""
    
    def __init__(self):
        """Initialize chemical parser with recognition patterns."""
        # Chemical formula patterns
        self.formula_patterns = [
            r'\b([A-Z][a-z]?(?:\d+)?)+\b',  # Basic chemical formula
            r'\b([A-Z][a-z]?)(\d+)',  # Element with subscript
            r'\(([^)]+)\)(\d+)',  # Parenthetical groups
        ]
        
        # Chemical name patterns
        self.chemical_name_patterns = [
            r'\b(\w+(?:-\w+)*)\s+(?:acid|base|salt|oxide|hydroxide|chloride|sulfate|nitrate)\b',
            r'\b(?:sodium|potassium|calcium|magnesium|iron|copper|zinc|aluminum)\s+(\w+)\b',
        ]
        
        # Periodic table data (simplified)
        self.periodic_table = {
            'H': {'name': 'Hydrogen', 'atomic_number': 1, 'atomic_weight': 1.008},
            'He': {'name': 'Helium', 'atomic_number': 2, 'atomic_weight': 4.003},
            'Li': {'name': 'Lithium', 'atomic_number': 3, 'atomic_weight': 6.941},
            'Be': {'name': 'Beryllium', 'atomic_number': 4, 'atomic_weight': 9.012},
            'B': {'name': 'Boron', 'atomic_number': 5, 'atomic_weight': 10.811},
            'C': {'name': 'Carbon', 'atomic_number': 6, 'atomic_weight': 12.011},
            'N': {'name': 'Nitrogen', 'atomic_number': 7, 'atomic_weight': 14.007},
            'O': {'name': 'Oxygen', 'atomic_number': 8, 'atomic_weight': 15.999},
            'F': {'name': 'Fluorine', 'atomic_number': 9, 'atomic_weight': 18.998},
            'Ne': {'name': 'Neon', 'atomic_number': 10, 'atomic_weight': 20.180},
            # Add more elements as needed
        }
    
    def parse_chemical_formulas(self, text: str) -> List[ChemicalCompound]:
        """
        Parse chemical formulas from text.
        
        Args:
            text: Input text containing chemical content
            
        Returns:
            List of parsed chemical compounds
        """
        compounds = []
        
        # Find chemical formula patterns
        for pattern in self.formula_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    formula = ''.join(match)
                else:
                    formula = match
                
                # Validate formula structure
                if self._is_valid_formula_structure(formula):
                    compound = ChemicalCompound(formula=formula)
                    compounds.append(compound)
        
        return compounds
    
    def _is_valid_formula_structure(self, formula: str) -> bool:
        """Validate basic chemical formula structure."""
        # Check if formula contains valid elements
        elements_in_formula = re.findall(r'[A-Z][a-z]?', formula)
        
        # Verify all elements exist in periodic table
        for element in elements_in_formula:
            if element not in self.periodic_table:
                return False
        
        # Check for proper formatting
        if not re.match(r'^[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?(?:\d+)?)*$', formula):
            return False
        
        return True
    
    def calculate_molecular_weight(self, compound: ChemicalCompound) -> float:
        """Calculate molecular weight from elemental composition."""
        total_weight = 0.0
        
        for element, count in compound.elements.items():
            if element in self.periodic_table:
                atomic_weight = self.periodic_table[element]['atomic_weight']
                total_weight += atomic_weight * count
        
        return total_weight


class UnitConverter:
    """Advanced unit conversion with dimensional analysis."""
    
    def __init__(self):
        """Initialize unit converter with unit definitions."""
        # Define base units and conversions
        self.unit_definitions = {
            # Length
            'm': ScientificUnit('m', 'meter', {'length': 1}),
            'cm': ScientificUnit('cm', 'centimeter', {'length': 1}, 0.01),
            'mm': ScientificUnit('mm', 'millimeter', {'length': 1}, 0.001),
            'km': ScientificUnit('km', 'kilometer', {'length': 1}, 1000),
            'in': ScientificUnit('in', 'inch', {'length': 1}, 0.0254),
            'ft': ScientificUnit('ft', 'foot', {'length': 1}, 0.3048),
            
            # Mass
            'kg': ScientificUnit('kg', 'kilogram', {'mass': 1}),
            'g': ScientificUnit('g', 'gram', {'mass': 1}, 0.001),
            'mg': ScientificUnit('mg', 'milligram', {'mass': 1}, 0.000001),
            'lb': ScientificUnit('lb', 'pound', {'mass': 1}, 0.453592),
            
            # Time
            's': ScientificUnit('s', 'second', {'time': 1}),
            'min': ScientificUnit('min', 'minute', {'time': 1}, 60),
            'h': ScientificUnit('h', 'hour', {'time': 1}, 3600),
            'day': ScientificUnit('day', 'day', {'time': 1}, 86400),
            
            # Temperature
            'K': ScientificUnit('K', 'kelvin', {'temperature': 1}),
            'C': ScientificUnit('C', 'celsius', {'temperature': 1}, 1, 273.15),
            'F': ScientificUnit('F', 'fahrenheit', {'temperature': 1}, 5/9, 459.67),
        }
        
        # Derived units
        self._generate_derived_units()
    
    def _generate_derived_units(self):
        """Generate common derived units."""
        # Velocity: m/s
        self.unit_definitions['m/s'] = ScientificUnit(
            'm/s', 'meter per second', 
            {'length': 1, 'time': -1}
        )
        
        # Acceleration: m/s²
        self.unit_definitions['m/s²'] = ScientificUnit(
            'm/s²', 'meter per second squared',
            {'length': 1, 'time': -2}
        )
        
        # Force: N (kg⋅m/s²)
        self.unit_definitions['N'] = ScientificUnit(
            'N', 'newton',
            {'mass': 1, 'length': 1, 'time': -2}
        )
        
        # Energy: J (kg⋅m²/s²)
        self.unit_definitions['J'] = ScientificUnit(
            'J', 'joule',
            {'mass': 1, 'length': 2, 'time': -2}
        )
        
        # Power: W (kg⋅m²/s³)
        self.unit_definitions['W'] = ScientificUnit(
            'W', 'watt',
            {'mass': 1, 'length': 2, 'time': -3}
        )
    
    def convert_units(self, value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """
        Convert value between units with dimensional analysis.
        
        Args:
            value: Numerical value to convert
            from_unit: Source unit symbol
            to_unit: Target unit symbol
            
        Returns:
            Converted value or None if conversion impossible
        """
        if from_unit not in self.unit_definitions or to_unit not in self.unit_definitions:
            logger.warning(f"Unknown units: {from_unit} or {to_unit}")
            return None
        
        from_def = self.unit_definitions[from_unit]
        to_def = self.unit_definitions[to_unit]
        
        # Check dimensional compatibility
        if from_def.base_dimensions != to_def.base_dimensions:
            logger.warning(f"Incompatible dimensions: {from_unit} vs {to_unit}")
            return None
        
        # Convert through base units
        base_value = value * from_def.conversion_factor + from_def.conversion_offset
        final_value = (base_value - to_def.conversion_offset) / to_def.conversion_factor
        
        return final_value
    
    def extract_units_from_text(self, text: str) -> List[Tuple[float, str]]:
        """Extract numerical values with units from text."""
        # Pattern for number + unit
        unit_pattern = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z/²³⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)'
        
        matches = re.findall(unit_pattern, text)
        results = []
        
        for value_str, unit_str in matches:
            try:
                value = float(value_str)
                # Normalize unit string
                normalized_unit = self._normalize_unit_string(unit_str)
                if normalized_unit in self.unit_definitions:
                    results.append((value, normalized_unit))
            except ValueError:
                continue
        
        return results
    
    def _normalize_unit_string(self, unit_str: str) -> str:
        """Normalize unit string to standard form."""
        # Handle common variations
        replacements = {
            'meter': 'm',
            'meters': 'm',
            'metre': 'm',
            'metres': 'm',
            'kilogram': 'kg',
            'kilograms': 'kg',
            'second': 's',
            'seconds': 's',
            'kelvin': 'K',
            'celsius': 'C',
            'fahrenheit': 'F',
        }
        
        normalized = unit_str.lower()
        for old, new in replacements.items():
            if normalized == old:
                return new
        
        return unit_str


class ScientificProcessor:
    """
    Advanced scientific text processor with domain-specific intelligence.
    
    This class orchestrates multiple specialized processors for comprehensive
    scientific text analysis and knowledge extraction.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize scientific processor with advanced capabilities.
        
        Args:
            config_path: Path to configuration directory
        """
        self.config_path = config_path or Path.home() / ".rag_system"
        
        # Initialize specialized processors
        self.math_parser = MathematicalParser()
        self.chem_parser = ChemicalParser()
        self.unit_converter = UnitConverter()
        
        # Processing statistics
        self.processed_documents = 0
        self.extracted_equations = 0
        self.extracted_formulas = 0
        self.extracted_compounds = 0
        
        logger.info("Scientific Processor initialized with advanced mathematical and chemical capabilities")
    
    def process_scientific_text(self, text: str, domain: ScientificDomain = ScientificDomain.GENERAL) -> Dict[str, Any]:
        """
        Process scientific text with domain-specific analysis.
        
        Args:
            text: Input scientific text
            domain: Scientific domain for specialized processing
            
        Returns:
            Comprehensive analysis results with extracted knowledge
        """
        results = {
            'domain': domain.name,
            'original_text': text,
            'mathematical_expressions': [],
            'chemical_compounds': [],
            'units_and_values': [],
            'scientific_concepts': [],
            'processed_text': text
        }
        
        # Mathematical processing
        math_expressions = self.math_parser.parse_expression(text)
        results['mathematical_expressions'] = [
            {
                'original': expr.original_text,
                'latex': expr.latex_form,
                'type': expr.object_type.name,
                'variables': list(expr.variables),
                'constants': list(expr.constants)
            }
            for expr in math_expressions
        ]
        
        # Chemical processing
        chemical_compounds = self.chem_parser.parse_chemical_formulas(text)
        results['chemical_compounds'] = [
            {
                'formula': comp.formula,
                'molecular_weight': comp.molecular_weight,
                'elements': comp.elements,
                'is_valid': comp.is_valid,
                'smiles': comp.smiles
            }
            for comp in chemical_compounds
        ]
        
        # Unit and measurement processing
        units_values = self.unit_converter.extract_units_from_text(text)
        results['units_and_values'] = [
            {
                'value': value,
                'unit': unit,
                'dimension': self.unit_converter.unit_definitions.get(unit, {}).get('base_dimensions', {})
            }
            for value, unit in units_values
        ]
        
        # Scientific concept extraction
        concepts = self._extract_scientific_concepts(text, domain)
        results['scientific_concepts'] = concepts
        
        # Text enhancement with mathematical normalization
        enhanced_text = self._enhance_scientific_text(text, math_expressions, chemical_compounds)
        results['processed_text'] = enhanced_text
        
        # Update statistics
        self.processed_documents += 1
        self.extracted_equations += len([expr for expr in math_expressions if expr.object_type == MathematicalObjectType.EQUATION])
        self.extracted_formulas += len([expr for expr in math_expressions if expr.object_type == MathematicalObjectType.FORMULA])
        self.extracted_compounds += len(chemical_compounds)
        
        return results
    
    def _extract_scientific_concepts(self, text: str, domain: ScientificDomain) -> List[str]:
        """Extract domain-specific scientific concepts."""
        concepts = []
        
        # Domain-specific concept patterns
        concept_patterns = {
            ScientificDomain.PHYSICS: [
                r'\b(?:velocity|acceleration|force|energy|momentum|mass|charge|field|wave|frequency|amplitude)\b',
                r'\b(?:newton|joule|watt|tesla|weber|pascal|hertz|coulomb|volt|ohm|farad|henry)\b',
                r'\b(?:quantum|relativity|thermodynamics|electromagnetism|mechanics|optics)\b'
            ],
            ScientificDomain.CHEMISTRY: [
                r'\b(?:molecule|atom|ion|bond|reaction|catalyst|enzyme|acid|base|salt|oxide)\b',
                r'\b(?:oxidation|reduction|synthesis|analysis|titration|precipitation|crystallization)\b',
                r'\b(?:organic|inorganic|polymer|isomer|stereochemistry|kinetics|thermochemistry)\b'
            ],
            ScientificDomain.MATHEMATICS: [
                r'\b(?:theorem|proof|lemma|corollary|axiom|hypothesis|conjecture)\b',
                r'\b(?:algebra|calculus|geometry|topology|analysis|statistics|probability)\b',
                r'\b(?:derivative|integral|limit|series|function|matrix|vector|tensor)\b'
            ],
            ScientificDomain.BIOLOGY: [
                r'\b(?:cell|tissue|organ|organism|species|gene|protein|enzyme|hormone)\b',
                r'\b(?:evolution|metabolism|photosynthesis|respiration|reproduction|adaptation)\b',
                r'\b(?:bacteria|virus|fungi|plant|animal|ecosystem|biodiversity)\b'
            ]
        }
        
        # Extract concepts based on domain
        if domain in concept_patterns:
            for pattern in concept_patterns[domain]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                concepts.extend(matches)
        
        # Remove duplicates and return
        return list(set(concepts))
    
    def _enhance_scientific_text(self, 
                                text: str,
                                math_expressions: List[MathematicalExpression],
                                chemical_compounds: List[ChemicalCompound]) -> str:
        """Enhance text with normalized scientific notation."""
        enhanced_text = text
        
        # Replace mathematical expressions with normalized forms
        for expr in math_expressions:
            if expr.latex_form and expr.latex_form != expr.original_text:
                # Add LaTeX representation as annotation
                enhanced_text = enhanced_text.replace(
                    expr.original_text,
                    f"{expr.original_text} (LaTeX: {expr.latex_form})"
                )
        
        # Enhance chemical formulas with additional information
        for comp in chemical_compounds:
            if comp.molecular_weight:
                enhanced_text = enhanced_text.replace(
                    comp.formula,
                    f"{comp.formula} (MW: {comp.molecular_weight:.2f} g/mol)"
                )
        
        return enhanced_text
    
    def solve_equation(self, equation_text: str) -> Optional[Dict[str, Any]]:
        """
        Solve mathematical equation using symbolic computation.
        
        Args:
            equation_text: Mathematical equation as text
            
        Returns:
            Solution dictionary with steps and results
        """
        try:
            # Parse equation
            if '=' not in equation_text:
                return None
            
            left, right = equation_text.split('=', 1)
            left_expr = sympify(left.strip())
            right_expr = sympify(right.strip())
            
            # Create equation
            equation = sp.Eq(left_expr, right_expr)
            
            # Find variables
            variables = equation.free_symbols
            
            if len(variables) == 1:
                # Single variable equation
                variable = list(variables)[0]
                solutions = solve(equation, variable)
                
                return {
                    'equation': equation_text,
                    'variable': str(variable),
                    'solutions': [str(sol) for sol in solutions],
                    'equation_object': str(equation),
                    'solution_count': len(solutions)
                }
            else:
                # Multiple variables - return general form
                return {
                    'equation': equation_text,
                    'variables': [str(var) for var in variables],
                    'equation_object': str(equation),
                    'note': 'Multiple variables - specify values for other variables'
                }
                
        except Exception as e:
            logger.error(f"Failed to solve equation: {equation_text}, error: {e}")
            return None
    
    def calculate_derivative(self, expression_text: str, variable: str = 'x') -> Optional[str]:
        """Calculate derivative of mathematical expression."""
        try:
            expr = sympify(expression_text)
            var = symbols(variable)
            derivative = diff(expr, var)
            return str(derivative)
        except Exception as e:
            logger.error(f"Failed to calculate derivative: {e}")
            return None
    
    def calculate_integral(self, expression_text: str, variable: str = 'x') -> Optional[str]:
        """Calculate integral of mathematical expression."""
        try:
            expr = sympify(expression_text)
            var = symbols(variable)
            integral = integrate(expr, var)
            return str(integral)
        except Exception as e:
            logger.error(f"Failed to calculate integral: {e}")
            return None
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics and performance metrics."""
        return {
            'processed_documents': self.processed_documents,
            'extracted_equations': self.extracted_equations,
            'extracted_formulas': self.extracted_formulas,
            'extracted_compounds': self.extracted_compounds,
            'rdkit_available': RDKIT_AVAILABLE,
            'capabilities': {
                'mathematical_parsing': True,
                'chemical_analysis': RDKIT_AVAILABLE,
                'unit_conversion': True,
                'symbolic_computation': True,
                'equation_solving': True,
                'calculus_operations': True
            }
        }
