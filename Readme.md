SaQC - `readthedocs-redirect` branch
====================================

This (protected) branch redirects https://saqc.readthedocs.io/ to our own gitlab pages (https://rdm-software.pages.ufz.de/saqc/)

Keep up the great work! :thumbsup:


### How it works

Because there is no general redirect for readthedocs (RTD) we have this branch wich provides a simnple redirect via sphinx (see `index.rst`). 
This works fine as long we go the the *exact* URL or to the URL with the `/en/latest` postfix (or `/pages/`), but it will not work with any other
pre/postfix. 

There are some configurable redirects via the RTD configugration (see https://readthedocs.org/dashboard/saqc/redirects/) but they redirect 
`OLD-URL/path` to `NEW-URL/path`, but the path is almost alwys not available on our gitlab-pages. 

The hacky solution prefix the `path` with a `#`. 

###### example
```
saqc.readthedocs.io/some/foo -> https://rdm-software.pages.ufz.de/saqc/#/some/foo
```

Thats not pretty but it works :innocent:
