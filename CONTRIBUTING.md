# Contributing to World Simulation

Thank you for your interest in contributing to the World Simulation project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

Before creating an issue, please:
1. Check if the issue already exists
2. Use the issue templates when available
3. Provide clear steps to reproduce the problem
4. Include relevant system information (OS, Python version, etc.)

### Suggesting Enhancements

We welcome suggestions for new features! Please:
1. Check existing issues and discussions first
2. Describe the enhancement clearly
3. Explain why it would be valuable
4. Consider implementation complexity

### Code Contributions

#### Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/World_Simulation.git
   cd World_Simulation
   ```

2. **Set up development environment**
   ```bash
   pip install -r requirements.txt
   # Optional: Install development dependencies
   pip install pytest black flake8 mypy
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Development Guidelines

**Code Style**
- Follow PEP 8 Python style guidelines
- Use type hints for function parameters and return values
- Write clear, descriptive variable and function names
- Add docstrings for classes and public methods

**Testing**
- Write tests for new functionality
- Ensure existing tests still pass
- Aim for good test coverage

**Documentation**
- Update README.md if adding new features
- Add docstrings for new functions/classes
- Update inline comments for complex logic

#### Project Structure

```
World_Simulation/
├── world_sim/                  # Core simulation package
│   ├── core/                  # Core simulation logic
│   │   ├── entities/          # Creature and trait definitions
│   │   └── behaviors/         # Behavior strategies
│   ├── logging/               # Data logging and export
│   └── ui/                    # User interfaces (CLI and web)
├── main.py                    # CLI entry point
├── app.py                     # Web interface entry point
└── requirements.txt           # Dependencies
```

#### Areas for Contribution

**Core Simulation**
- New creature behaviors
- Additional genetic traits
- Environmental factors
- Performance optimizations

**User Interface**
- Terminal UI improvements
- Web dashboard enhancements
- New visualization options
- Better configuration options

**Data & Analytics**
- New metrics and statistics
- Export formats
- Data analysis tools
- Performance monitoring

**Documentation**
- Code documentation
- User guides
- Tutorial content
- Examples and demos

#### Submitting Changes

1. **Test your changes**
   ```bash
   python main.py  # Test CLI interface
   streamlit run app.py  # Test web interface
   ```

2. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

3. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request**
   - Use a clear, descriptive title
   - Reference any related issues
   - Provide a detailed description of changes
   - Include screenshots for UI changes

## 📋 Pull Request Guidelines

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] New features have appropriate tests
- [ ] Documentation is updated
- [ ] All tests pass
- [ ] No merge conflicts

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe how you tested your changes

## Screenshots (if applicable)
Add screenshots for UI changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**
   - Operating System
   - Python version
   - Package versions

2. **Steps to Reproduce**
   - Clear, numbered steps
   - Expected vs actual behavior
   - Error messages (if any)

3. **Additional Context**
   - Screenshots or logs
   - Related issues
   - Workarounds (if any)

## 💡 Feature Requests

When suggesting features:

1. **Problem Description**
   - What problem does this solve?
   - Who would benefit from this feature?

2. **Proposed Solution**
   - How should it work?
   - Any implementation ideas?

3. **Alternatives Considered**
   - Other approaches you've thought about
   - Why this approach is preferred

## 📚 Development Resources

- [Python Style Guide (PEP 8)](https://pep8.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Rich Documentation](https://rich.readthedocs.io/)

## 🏷️ Labels

We use labels to categorize issues and PRs:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information is requested

## 📞 Getting Help

- **GitHub Discussions**: For questions and general discussion
- **Issues**: For bug reports and feature requests
- **Pull Requests**: For code contributions

## 🙏 Recognition

Contributors will be recognized in:
- README.md acknowledgments
- Release notes
- Project documentation

Thank you for contributing to World Simulation! 🎉
