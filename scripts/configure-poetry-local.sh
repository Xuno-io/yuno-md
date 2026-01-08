#!/bin/bash
# Configure Poetry to use Google Artifact Registry for local development

set -eo pipefail

# Configuration
PROJECT_ID="yunoai-1679262920707"
REGION="us-central1"
REPOSITORY="xuno-pypi"

echo "🔧 Configuring Poetry for Google Artifact Registry..."

# Verify gcloud CLI
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI is not installed."
    echo "   Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Verify authentication
if ! gcloud auth list --filter="status:ACTIVE" --format="value(account)" | head -n1 | grep -q "@"; then
    echo "❌ Error: No active gcloud authentication."
    echo "   Run: gcloud auth login"
    exit 1
fi

# Get access token for Artifact Registry
echo "🔑 Getting access token..."
ACCESS_TOKEN=$(gcloud auth print-access-token)

if [ -z "$ACCESS_TOKEN" ]; then
    echo "❌ Error: Could not get access token."
    exit 1
fi

# Repository URLs
REPO_URL="https://${REGION}-python.pkg.dev/${PROJECT_ID}/${REPOSITORY}/"
SIMPLE_URL="${REPO_URL}simple/"

# Configure Poetry repository
echo "📦 Configuring Poetry repository..."
poetry config repositories.xuno $REPO_URL

# Configure authentication using oauth2accesstoken as username
poetry config http-basic.xuno oauth2accesstoken $ACCESS_TOKEN

echo ""
echo "✅ Poetry configured for Artifact Registry!"
echo ""
echo "📌 Repository URL: $REPO_URL"
echo ""
echo "You can now:"
echo "  1. Publish: poetry publish -r xuno"
echo "  2. Install: poetry add xuno-components --source xuno"
echo ""
echo "⚠️  Note: The access token expires in ~1 hour."
echo "   Re-run this script when you need to refresh it."