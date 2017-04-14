/*
Pulls out the first device and os that a patient used the app with.
Applies filters to remove users that were referred to the app directly by a clinician
*/
SELECT 
    c.user_id, dt.os_type, dt.device_form, c.lead_source, u.is_demo, s.type
FROM
    ct_customer.customers c
        JOIN
    constant_therapy.users u ON c.user_id = u.id
        JOIN
	constant_therapy.sessions s ON c.user_id = s.patient_id
		JOIN
    constant_therapy.user_events ue ON ue.user_id = c.user_id
        AND ue.event_type_id = 32 -- when they log in
        AND ue.event_sub_type = 'clientHardwareType' -- when the client
        AND event_data like 'null|%' -- replacement
        LEFT JOIN
    constant_therapy.device_types dt ON right(ue.event_data,
        length(ue.event_data) - 5) = dt.client_hardware_type
WHERE
	c.lead_source != 'Clinician_Setup'
	AND u.is_demo != 1
	AND s.type = 'SCHEDULED' -- changed
LIMIT 1000000000;