

set -ex



pip check
pytest -n auto -k "not (test_der_x509_certificate_extensions[x509/PKITS_data/certs/ValidcRLIssuerTest28EE.crt] or test_x509_csr_extensions or test_no_leak_free or test_no_leak_no_malloc or test_leak or test_load_pkcs12_key_and_certificates[pkcs12/cert-key-aes256cbc.p12] or test_create_certificate_with_extensions or test_ec_derive_private_key or test_ec_private_numbers_private_key or test_create_ocsp_request or test_write_pkcs12_key_and_certificates or test_errors or test_load_pkcs12_key_and_certificates[pkcs12/cert-aes256cbc-no-key.p12] or test_ec_private_numbers_private_key or test_pem_x509_certificate_extensions[x509/cryptography.io.pem] or test_create_crl_with_idp or test_no_leak_gc or test_x25519_pubkey_from_private_key)"
exit 0
