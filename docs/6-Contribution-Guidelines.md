# 6. Contribution Guidelines

We welcome contributions to HiGP. This document provides guidelines for contributing to the project, filing issues, and getting support.

## 6.1 How to Contribute

### 6.1.1 Types of Contributions

We appreciate various types of contributions, including:

- **Bug fixes**: Help us identify and fix bugs in the code
- **New features**: Implement new kernels, solvers, or GP methods
- **Documentation improvements**: Enhance user guides, API documentation, or examples
- **Performance optimizations**: Improve the speed or memory efficiency of existing code
- **Test cases**: Add unit tests or integration tests to improve code coverage
- **Example notebooks**: Create new examples demonstrating HiGP's capabilities

### 6.1.2 Getting Started

1. **Fork the repository**: Create your own fork of the [HiGP repository](https://github.com/huanghua1994/HiGP)
2. **Clone your fork**: `git clone https://github.com/YOUR_USERNAME/HiGP.git`
3. **Create a branch**: `git checkout -b your-feature-branch`
4. **Set up your development environment**: Follow the instructions in [Section 4.2 of the Developer Information](4-Developer-information.md#42-building-higp-on-local-machine)

### 6.1.3 Making Changes

1. **Write clean code**: Follow the existing code style and conventions
   - For C++ code: Use consistent indentation, meaningful variable names, and add comments for complex logic
   - For Python code: Follow PEP 8 style guidelines
2. **Add tests**: If you add new functionality, include appropriate unit tests in the `cpp-tests/` or `py-tests/` directories
3. **Update documentation**: If your changes affect the user interface or add new features, update the relevant documentation files in the `docs/` directory
4. **Test your changes**:
   - Run C++ unit tests: See Section 4.5 of the [Developer Information](4-Developer-information.md#45-debugging-the-cc-source-code-on-local-machine)
   - Run Python tests: `pytest py-tests/`

### 6.1.4 Submitting a Pull Request

1. **Commit your changes**: Write clear, descriptive commit messages

   ```bash
   git add .
   git commit -m "Add feature: brief description of your changes"
   ```

2. **Push to your fork**: `git push origin your-feature-branch`
3. **Create a pull request**: Go to the original HiGP repository and create a pull request from your branch
4. **Describe your changes**: In the pull request description, explain:
   - What changes you made and why
   - Any related issues (reference with `#issue_number`)
   - How you tested your changes
5. **Address review comments**: Be responsive to feedback from maintainers

### 6.1.5 Code Review Process

- A maintainer will review your pull request
- You may be asked to make changes or provide additional information
- Once approved, a maintainer will merge your contribution
- Please be patient; reviews may take some time depending on the complexity of the changes

## 6.2 Filing Issues

### 6.2.1 Before Filing an Issue

Before creating a new issue, please:

1. **Search existing issues**: Check if your issue has already been reported
2. **Check the documentation**: Review the [documentation](0-List-of-contents.md) to ensure you're using HiGP correctly
3. **Try the latest version**: Ensure you're using the most recent version of HiGP: `pip install --upgrade higp`

### 6.2.2 How to Report a Bug

When reporting a bug, please include:

1. **Clear title**: A concise description of the problem
2. **HiGP version**: The version you're using (`import higp; print(higp.__version__)`)
3. **Environment details**:
   - Operating system (e.g., Ubuntu 22.04, macOS 13, Windows 11)
   - Python version
   - NumPy and PyTorch versions
   - BLAS library (if known)
4. **Minimal reproducible example**: A short code snippet that demonstrates the bug
5. **Expected behavior**: What you expected to happen
6. **Actual behavior**: What actually happened, including any error messages or stack traces
7. **Additional context**: Any other relevant information

### 6.2.3 Requesting Features

When requesting a new feature:

1. **Check existing issues**: See if the feature has already been requested
2. **Describe the feature**: Clearly explain what you would like to see added
3. **Explain the use case**: Describe why this feature would be useful
4. **Suggest implementation** (optional): If you have ideas about how to implement the feature, share them

## 6.3 Getting Support

### 6.3.1 Questions and Discussions

If you have questions about using HiGP:

1. **Check the documentation**: Start with the [Basic Usage](1-Basic-usage-of-HiGP.md) and [Advanced Usage](2-Advanced-usage-of-HiGP.md) guides
2. **Review the examples**: Look at the Jupyter notebooks in the `py-demos/` directory
3. **Search existing issues**: Your question may have been answered before
4. **Open a GitHub issue**: If you can't find an answer, create an issue with the "question" label

### 6.3.2 Contact Information

For direct support, you can contact the main developers:

- **Hua Huang** (<huangh1994@outlook.com>)
- **Tianshi Xu** (<tianshi.xu@emory.edu>)

The maintainers contribute to this project in their available time. We will do our best to respond to issues and questions as quickly as possible.

## 6.4 License

By contributing to HiGP, you agree that your contributions will be licensed under the same license as the project. See the `LICENSE` file in the repository root for details.

---

Thank you for contributing to HiGP. Your efforts help make this project better for everyone.
