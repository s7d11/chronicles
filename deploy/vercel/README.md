# Volume II · Optional Custom Domain (Namecheap Stellar DNS) & Cron Notes

1. **Primary Hosting (Zero Cost, Static-First)**:
   GitHub Pages serves the entire `site/` broadsheet (`https://s7d11.github.io/chronicles/`), while live updates stream via Cloud Firestore Web SDK `onSnapshot` from `chronicles-live` (and static replay snapshots in `site/data/vol2_*.json`).

2. **Optional Namecheap Stellar Custom Subdomain (`chronicle.<yourdomain>`)**:
   - In GitHub Repository Settings → **Pages** → **Custom domain**, enter `chronicle.<yourdomain>`.
   - In **Namecheap cPanel → Zone Editor**, add:
     - **Host**: `chronicle` (or `@` apex with GitHub A records `185.199.108–111.153`)
     - **Type**: `CNAME`
     - **Target**: `s7d11.github.io`
   - Enforce **HTTPS** once DNS propagates. Zero VPS server processes are required on Stellar shared hosting!
