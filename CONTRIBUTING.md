# Contributing to Manipulador

Thank you for your interest in contributing to Manipulador! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

### Our Standards

- **Be respectful and inclusive**: Treat all contributors with respect regardless of their background
- **Focus on constructive feedback**: Provide helpful, actionable feedback in reviews and discussions
- **Collaborate openly**: Share knowledge and help others learn and grow
- **Prioritize safety**: This is a security research tool - ensure contributions maintain responsible disclosure practices

## How to Contribute

### Reporting Issues

1. **Check existing issues** first to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information**:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, GPU info)
   - Relevant logs or error messages

### Submitting Code Changes

#### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/your-username/Manipulador-OSS.git
cd Manipulador-OSS

# Add upstream remote
git remote add upstream https://github.com/your-org/Manipulador-OSS.git
```

#### 2. Create a Feature Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name
```

#### 3. Make Your Changes

- Follow the coding style (see Style Guide below)
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

#### 4. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a clear message
git commit -m "Add feature: brief description of changes"
```

#### 5. Push and Create Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Create a pull request on GitHub
```

### Pull Request Guidelines

- **Clear title and description**: Explain what the PR does and why
- **Link related issues**: Reference any issues your PR addresses
- **Small, focused changes**: Keep PRs manageable and focused on a single feature/fix
- **Update tests and docs**: Ensure tests pass and documentation is updated
- **Follow the template**: Use the PR template when provided

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA-capable GPU (recommended)

### Environment Setup

```bash
# Create virtual environment
python -m venv manipulador-dev
source manipulador-dev/bin/activate  # On Windows: manipulador-dev\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agents.py

# Run with coverage
pytest --cov=manipulador --cov-report=html
```

### Code Style Guide

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **isort**: Import sorting

```bash
# Format code
black .

# Check linting
flake8 .

# Sort imports
isort .

# Or run all pre-commit hooks
pre-commit run --all-files
```

#### Style Guidelines

1. **PEP 8 compliance**: Follow Python's official style guide
2. **Clear naming**: Use descriptive variable and function names
3. **Type hints**: Add type hints for function parameters and return values
4. **Docstrings**: Document all public functions and classes
5. **Comments**: Explain complex logic and design decisions

Example:

```python
from typing import Dict, List, Any

def analyze_behavior_selection(
    behavior_name: str, 
    model_characteristics: Dict[str, Any]
) -> Dict[str, float]:
    """
    Analyze behavior and return method selection scores.
    
    Args:
        behavior_name: Name of the behavior to analyze
        model_characteristics: Target model metadata
        
    Returns:
        Dictionary mapping method names to selection scores
        
    Raises:
        ValueError: If behavior_name is empty or invalid
    """
    if not behavior_name.strip():
        raise ValueError("Behavior name cannot be empty")
    
    # Implementation here...
    return {"DirectRequest": 0.3, "FewShot": 0.7, "GPTFuzz": 0.5}
```

### Testing Guidelines

#### Test Structure

```
tests/
├── __init__.py
├── test_agents/
│   ├── test_v9_generic.py
│   └── test_v9_2step.py
├── test_baselines/
│   ├── test_directrequest.py
│   ├── test_fewshot.py
│   └── test_gptfuzz.py
├── test_scripts/
│   └── test_evaluation.py
└── fixtures/
    └── sample_data.py
```

#### Writing Tests

1. **Use pytest**: Leverage pytest features and fixtures
2. **Descriptive names**: Test function names should describe what they test
3. **Test edge cases**: Include boundary conditions and error cases
4. **Mock external dependencies**: Don't rely on external APIs or large models in tests
5. **Fast execution**: Keep unit tests quick to run

Example:

```python
import pytest
from unittest.mock import Mock, patch
from manipulador.agents.v9_generic import GenericV9BehaviorAttackSelector

class TestGenericV9BehaviorAttackSelector:
    
    @pytest.fixture
    def agent(self):
        return GenericV9BehaviorAttackSelector()
    
    def test_analyze_behavior_basic(self, agent):
        """Test basic behavior analysis functionality."""
        behavior_data = {"behavior": "test behavior", "category": "test"}
        model_data = {"safety_level": "medium", "size_category": "small"}
        
        result = agent.analyze_behavior_generic(
            "test_behavior", behavior_data, "test_model", model_data
        )
        
        assert "selected_method" in result
        assert result["selected_method"] in ["DirectRequest", "FewShot", "GPTFuzz"]
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
    
    def test_analyze_behavior_empty_input(self, agent):
        """Test behavior analysis with invalid input."""
        with pytest.raises(ValueError):
            agent.analyze_behavior_generic("", {}, "test_model", {})
```

## Documentation

### Types of Documentation

1. **Code documentation**: Docstrings and inline comments
2. **User guides**: How-to guides and tutorials (in `docs/`)
3. **API reference**: Generated from docstrings
4. **Examples**: Working code examples (in `examples/`)

### Writing Documentation

- **Clear and concise**: Use simple language and short sentences
- **Include examples**: Show practical usage
- **Update with changes**: Keep docs in sync with code changes
- **Test examples**: Ensure code examples actually work

## Security Considerations

Given that Manipulador is a security research tool, special considerations apply:

### Responsible Disclosure

- **Report security issues privately**: Don't create public issues for vulnerabilities
- **Follow coordinated disclosure**: Work with maintainers on timing
- **Provide sufficient detail**: Include reproduction steps and impact assessment

### Code Safety

- **Review attack methods carefully**: Ensure they're used for legitimate research
- **Limit harmful outputs**: Implement safeguards against misuse  
- **Document risks**: Clearly explain potential dangers and mitigations

### Research Ethics

- **Follow ethical guidelines**: Ensure research benefits the community
- **Respect model owners**: Don't publish specific model vulnerabilities without permission
- **Provide defenses**: Include mitigation strategies when possible

## Release Process

### Version Numbering

We use semantic versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Checklist

1. Update version numbers
2. Update CHANGELOG.md
3. Run full test suite
4. Update documentation
5. Create release notes
6. Tag release
7. Deploy to package repositories

## Community

### Getting Help

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bug reports and feature requests

### Stay Connected

- **Star the repository**: Stay updated with releases
- **Watch for issues**: Help others with problems you've solved
- **Share your work**: Let us know how you're using Manipulador
