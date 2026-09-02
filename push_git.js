const { execSync } = require('child_process');
const path = require('path');

const cwd = path.join(__dirname);

try {
    console.log('Staging files...');
    execSync('git add .gitignore README.md *.bmp recorded_demo/*.bmp', { cwd, stdio: 'inherit' });
    
    console.log('Committing...');
    execSync('git commit -m "docs: include LiDAR dashboard render frames and visual gallery in repo"', { cwd, stdio: 'inherit' });
    
    console.log('Pushing to GitHub...');
    execSync('git push origin main', { cwd, stdio: 'inherit' });
    
    console.log('SUCCESS: Photos pushed to repository!');
} catch (err) {
    console.error('Git execution output:', err.message);
}
