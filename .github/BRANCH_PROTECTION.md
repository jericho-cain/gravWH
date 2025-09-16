# Branch Protection Setup

To require tests to pass on PRs, you need to set up branch protection rules in GitHub:

## Steps to Enable Branch Protection:

1. Go to your repository on GitHub
2. Click on "Settings" tab
3. Click on "Branches" in the left sidebar
4. Click "Add rule" or edit the existing rule for `main`
5. Configure the following settings:

### Required Settings:
- ✅ **Require a pull request before merging**
- ✅ **Require status checks to pass before merging**
- ✅ **Require branches to be up to date before merging**
- ✅ **Require conversation resolution before merging**

### Status Checks to Require:
- `test (3.9)` - Python 3.9 tests
- `test (3.10)` - Python 3.10 tests  
- `test (3.11)` - Python 3.11 tests

### Additional Settings (Recommended):
- ✅ **Require linear history**
- ✅ **Include administrators**
- ✅ **Restrict pushes that create files that are larger than 100 MB**

## What This Achieves:

- **Tests must pass** before any PR can be merged
- **All Python versions** (3.9, 3.10, 3.11) must pass
- **Branches must be up to date** with main before merging
- **Conversations must be resolved** before merging
- **Linear history** prevents merge commits

## Alternative: Use GitHub CLI

You can also set this up via command line:

```bash
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["test (3.9)","test (3.10)","test (3.11)"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1}' \
  --field restrictions=null
```

This ensures code quality and prevents broken code from being merged into main.
