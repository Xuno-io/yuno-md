#!/bin/bash

# Script para renovar el token de CodeArtifact
echo "Renovando token de CodeArtifact..."

# Obtener nuevo token
CODEARTIFACT_AUTH_TOKEN=$(aws codeartifact get-authorization-token --domain xuno --region us-east-1 --query authorizationToken --output text)

if [ $? -eq 0 ]; then
    # Configurar Poetry con el nuevo token
    poetry config http-basic.xuno aws "$CODEARTIFACT_AUTH_TOKEN"
    echo "✅ Token renovado exitosamente"
else
    echo "❌ Error al obtener el token de CodeArtifact"
    exit 1
fi