# Contributing to Stock Price Movement Prediction System

Thank you for your interest in contributing to the Stock Price Movement Prediction System! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

We welcome contributions of all kinds:
- 🐛 Bug reports and fixes
- ✨ New features and enhancements
- 📚 Documentation improvements
- 🧪 Test coverage improvements
- 🎨 UI/UX improvements
- 📊 New stock symbols or data sources
- 🧠 Model improvements and new algorithms

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of machine learning and time series analysis
- Familiarity with TensorFlow/Keras and Streamlit

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/Stock-Price-Movement-Prediction-System.git
   cd Stock-Price-Movement-Prediction-System
   ```

2. **Set up development environment**
   ```bash
   # Using Make (recommended)
   make setup-dev
   
   # Or manually
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install pytest black isort flake8 mypy coverage
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number
   ```

4. **Verify setup**
   ```bash
   make check  # Runs tests and linting
   ```

## 📋 Development Guidelines

### Code Style

We follow Python best practices and use automated tools for consistency:

- **Formatting**: Black with 100 character line length
- **Import sorting**: isort with black profile
- **Linting**: flake8 with specific rules
- **Type hints**: mypy for static type checking

```bash
# Format code
make format

# Check code quality
make lint

# Run all checks
make check
```

### Code Standards

1. **Python Style**
   - Follow PEP 8 guidelines
   - Use type hints for function signatures
   - Write descriptive docstrings for classes and functions
   - Keep functions focused and small (< 50 lines when possible)

2. **Documentation**
   - Update README.md for new features
   - Add docstrings to all public functions and classes
   - Include inline comments for complex logic
   - Update API documentation when needed

3. **Testing**
   - Write unit tests for new functionality
   - Maintain test coverage above 80%
   - Include integration tests for major features
   - Test edge cases and error conditions

### Project Structure

```
src/
├── data/           # Data collection and preprocessing
├── features/       # Feature engineering
├── models/         # ML model implementations
├── evaluation/     # Model evaluation metrics
├── web/           # Streamlit web interface
└── utils/         # Utility functions

tests/             # Test files (mirror src structure)
docs/              # Documentation
examples/          # Usage examples
docker/            # Docker configuration
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_models.py -v

# Run with coverage
make test-coverage

# Run only unit tests
make test-unit
```

### Writing Tests

1. **Test Structure**
   - Place tests in `tests/` directory
   - Mirror the `src/` structure
   - Name test files as `test_*.py`

2. **Test Categories**
   - **Unit tests**: Test individual functions/classes
   - **Integration tests**: Test component interactions
   - **End-to-end tests**: Test complete workflows

3. **Test Examples**
   ```python
   import pytest
   from src.models.lstm_model import LSTMStockModel
   
   def test_lstm_model_creation():
       model = LSTMStockModel(input_shape=(30, 20), task_type="classification")
       assert model.task_type == "classification"
       assert not model.is_trained
   
   def test_model_training():
       # Test with sample data
       pass
   ```

## 📝 Pull Request Process

### Before Submitting

1. **Ensure code quality**
   ```bash
   make check  # Must pass all checks
   ```

2. **Update documentation**
   - Update README.md if needed
   - Add/update docstrings
   - Update CHANGELOG.md

3. **Test thoroughly**
   - Run full test suite
   - Test manually with web interface
   - Verify Docker build works

### Pull Request Guidelines

1. **Title and Description**
   - Use clear, descriptive titles
   - Reference related issues (#123)
   - Explain what changes were made and why
   - Include screenshots for UI changes

2. **Commit Messages**
   ```
   feat: add support for new technical indicators
   fix: resolve prediction index out of bounds error
   docs: update installation instructions
   test: add unit tests for feature engineering
   refactor: improve model training pipeline
   ```

3. **Review Process**
   - Address reviewer feedback promptly
   - Keep discussions constructive and professional
   - Update code based on suggestions
   - Ensure CI/CD checks pass

## 🐛 Bug Reports

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Try latest version** to see if bug is already fixed
3. **Gather information** about your environment

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- TensorFlow version: [e.g., 2.8.0]
- Browser: [e.g., Chrome 96] (for web interface issues)

**Additional Context**
- Error messages or logs
- Screenshots if applicable
- Any other relevant information
```

## ✨ Feature Requests

### Feature Request Template

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Use Case**
Explain why this feature would be useful.

**Proposed Solution**
Describe how you envision this feature working.

**Alternatives Considered**
Any alternative solutions you've considered.

**Additional Context**
Any other context, mockups, or examples.
```

## 🏷️ Issue Labels

We use labels to categorize issues:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested
- `wontfix`: This will not be worked on

## 🎯 Areas for Contribution

### High Priority
- 🧠 **Model Improvements**: New architectures, hyperparameter tuning
- 📊 **Data Sources**: Additional stock exchanges, crypto support
- 🔧 **Performance**: Optimization, caching, parallel processing
- 🧪 **Testing**: Increase test coverage, add integration tests

### Medium Priority
- 🎨 **UI/UX**: Streamlit interface improvements
- 📚 **Documentation**: Tutorials, API docs, examples
- 🐳 **DevOps**: CI/CD improvements, deployment automation
- 🌐 **Internationalization**: Multi-language support

### Good First Issues
- 📝 **Documentation**: Fix typos, improve clarity
- 🧹 **Code Cleanup**: Remove unused imports, improve comments
- 🎨 **UI Polish**: Small interface improvements
- 📊 **Stock Symbols**: Add new stocks to the database

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Review**: Tag maintainers for review help

## 🏆 Recognition

Contributors will be:
- Listed in the README.md contributors section
- Mentioned in release notes for significant contributions
- Invited to join the maintainers team for consistent contributors

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You

Thank you for contributing to the Stock Price Movement Prediction System! Your efforts help make this project better for everyone.

---

**Questions?** Feel free to open an issue or start a discussion. We're here to help! 🚀
