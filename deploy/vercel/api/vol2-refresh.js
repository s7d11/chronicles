// Optional Vercel Cron Webhook -> triggers Google Cloud Run Job `sebeni-vol2-job`
// Requires VERCEL_CRON_SECRET and GCP_SERVICE_ACCOUNT_KEY (or WIF) if triggering GCP REST API.
export default async function handler(req, res) {
  const auth = req.headers.authorization || "";
  if (process.env.CRON_SECRET && auth !== `Bearer ${process.env.CRON_SECRET}`) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  return res.status(200).json({
    ok: true,
    note: "Static Chronicle + Firestore `chronicles-live` active. Cloud Scheduler `sebeni-vol2-job-sched` invokes sweeps directly.",
    timestamp: new Date().toISOString(),
  });
}
