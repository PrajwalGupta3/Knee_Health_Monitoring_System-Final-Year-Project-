// Shared helper file used by pages
console.log("web static loaded");

// Helper function to handle network requests and JSON parsing errors
async function fetchJSON(url, opts) {
  try {
    const res = await fetch(url, opts);
    
    // If the server returns a 404 or 500, fetch doesn't throw an error, 
    // so we check res.ok manually if needed, or let the json parsing handle it.
    
    return await res.json();
  } catch(e) {
    console.error("Fetch error:", e);
    return { ok: false, error: "Invalid JSON response or Server Error" };
  }
}