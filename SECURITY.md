# Security Policy

## 🔒 Reporting Security Vulnerabilities

The OnCall AI Agent team takes security seriously. We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

### 📧 How to Report a Security Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **security@oncall-ai-agent.org**

Include the following information in your report:
- Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

### 🛡️ Security Response Process

1. **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours.

2. **Initial Assessment**: We will perform an initial assessment of the report within 5 business days.

3. **Investigation**: We will investigate and validate the vulnerability.

4. **Resolution**: We will develop and test a fix.

5. **Disclosure**: We will coordinate disclosure with you, including:
   - Timeline for public disclosure
   - Credit attribution (if desired)
   - CVE assignment (if applicable)

### ⏰ Expected Response Times

- **Initial Response**: Within 48 hours
- **Status Update**: Within 5 business days
- **Resolution**: Varies based on complexity, but we aim for 30 days for critical issues

## 🛡️ Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | ✅ Yes             |
| 0.x.x   | ❌ No (Beta)       |

## 🔐 Security Best Practices

### For Users

1. **Keep Updated**: Always use the latest version of OnCall AI Agent
2. **Secure Configuration**: 
   - Use environment variables for sensitive credentials
   - Never commit API keys or secrets to version control
   - Use strong, unique passwords and API keys
3. **Network Security**:
   - Run behind a reverse proxy with HTTPS
   - Implement proper firewall rules
   - Use VPN for remote access if needed
4. **Access Control**:
   - Enable authentication if processing sensitive data
   - Use principle of least privilege for service accounts
   - Regularly rotate API keys and credentials

### For Developers

1. **Code Security**:
   - Follow secure coding practices
   - Validate all inputs
   - Use parameterized queries
   - Implement proper error handling
2. **Dependencies**:
   - Keep dependencies updated
   - Regularly scan for vulnerable dependencies
   - Use tools like `safety` and `bandit`
3. **Testing**:
   - Include security tests in CI/CD
   - Test with various input scenarios
   - Perform regular security audits

## 🔍 Security Features

### Built-in Security Measures

- **Input Validation**: All user inputs are validated and sanitized
- **Rate Limiting**: Built-in protection against abuse and DoS attacks
- **Secure Headers**: HTTP security headers are set by default
- **Audit Logging**: Comprehensive logging of all actions
- **Credential Management**: Secure handling of API keys and secrets
- **CORS Protection**: Configurable CORS policies

### Optional Security Features

- **Authentication**: JWT-based authentication system
- **Authorization**: Role-based access control
- **Encryption**: TLS/SSL encryption for data in transit
- **Secret Management**: Integration with secret management systems

## 🚨 Known Security Considerations

### Data Handling
- **Log Data**: Logs may contain sensitive information - ensure proper access controls
- **LLM Interactions**: Be aware that data sent to external LLM providers may be logged
- **Code Repository Access**: Repository access tokens should be scoped appropriately

### Network Security
- **External APIs**: The agent makes requests to various external APIs
- **Cloud Provider Access**: Requires cloud provider credentials with appropriate permissions

### Deployment Security
- **Container Security**: Use trusted base images and scan for vulnerabilities
- **Environment Variables**: Secure storage and injection of environment variables
- **Network Policies**: Implement appropriate network segmentation

## 🛠️ Security Tools and Scanning

We use the following tools to maintain security:

- **Static Analysis**: `bandit`, `semgrep`
- **Dependency Scanning**: `safety`, `pip-audit`
- **Container Scanning**: `trivy`, `snyk`
- **Code Quality**: `sonarcloud`

## 📋 Security Checklist for Deployments

- [ ] All credentials stored securely (environment variables, secrets manager)
- [ ] HTTPS enabled with valid certificates
- [ ] Rate limiting configured appropriately
- [ ] Authentication enabled if processing sensitive data
- [ ] Firewall rules configured to restrict access
- [ ] Logging enabled and monitored
- [ ] Regular security updates scheduled
- [ ] Backup and recovery procedures in place
- [ ] Incident response plan documented

## 🏆 Recognition

We believe in recognizing security researchers who help improve our security. With your permission, we will:

- Credit you in our security advisories
- Add you to our security researchers hall of fame
- Provide a reference letter if requested

## 📞 Contact Information

- **Security Email**: security@oncall-ai-agent.org
- **General Contact**: hello@oncall-ai-agent.org
- **GitHub Issues**: For non-security related issues only

## 📚 Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CIS Controls](https://www.cisecurity.org/controls)

---

**Thank you for helping keep OnCall AI Agent and our users safe!** 🙏
