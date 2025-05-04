#!/bin/bash

# Install dependencies
npm install

# Create next-env.d.ts file if it doesn't exist
if [ ! -f next-env.d.ts ]; then
    echo "Creating next-env.d.ts..."
    echo '/// <reference types="next" />
/// <reference types="next/image-types/global" />

// NOTE: This file should not be edited
// see https://nextjs.org/docs/basic-features/typescript for more information.' > next-env.d.ts
fi

echo "Setup complete. Run 'npm run dev' to start the development server." 