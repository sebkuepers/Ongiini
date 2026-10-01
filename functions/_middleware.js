// Pages middleware: www.ongiini.ai → ongiini.ai (one canonical host for
// search engines and shared links). Everything else passes straight through.
export async function onRequest(context) {
  const url = new URL(context.request.url);
  if (url.hostname === "www.ongiini.ai") {
    url.hostname = "ongiini.ai";
    return Response.redirect(url.toString(), 301);
  }
  return context.next();
}
