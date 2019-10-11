---
layout: single
tags: AWS
categories: IT
---



# Setting up AWS Render Farm with Deadline


10 min [youtube](https://www.youtube.com/watch?v=niaQ1OWimoI)

Checklist/steps:
1.  [AWS Portal Setup][AWS_Portal], [AWS Portal Checklist][AWS_checklist]
  - [x] AWS Portal Asset Server
  - [ ] AWS Portal Link
  a. Did they install it when they installed the asset server?
  a. same installer as asset server
  a. AWS Portal [link](https://docs.thinkboxsoftware.com/products/deadline/10.0/1_User%20Manual/manual/aws-portal-installing.html#aws-portal-link)
  - [ ] Deadline Remote Connection Server
  - [ ] Connect Deadline to RCS [link][deadline_to_rcs]
  - [ ] Set Asset Server Local IP
1. Usage based licensing
  - [ ] Thinkbox Marketplace - create account
  - [ ] Purchase render time (50 hrs = 10 bucks)
  - email notification w/ _cloud license server url_, _activation code_
  - [ ] download certs from [customer portal](https://thinkbox.flexnetoperations.com/control/tnkb/login?nextURL=%2Fcontrol%2Ftnkb%2Fpurchases)
1. UBL Setup
  - [ ] Follow steps in this [link][UBL_Setup]
  - Deadline requires _cloud license server URL_ and _activation code_
  - AWS Portal needs _AWS Access Key_, _Secret Access Key_
1. Configure [Limits][deadline_limits]
  - [ ] maya, slave, use UBL, check 'UBL Application', click 'ok', Unlimited Limit

__Should now be able to submit jobs!__

1. Infrastructure, link on [AWS_checklist][AWS_checklist]
1. Spot fleet, also on [AWS_checklist][AWS_checklist]

common [troublehsooting](https://docs.thinkboxsoftware.com/products/deadline/10.0/1_User%20Manual/manual/aws-portal-troubleshooting.html)

alternate [link](https://aws.amazon.com/blogs/media/rendering-with-aws-portal-in-thinkbox-deadline/) for submitting jobs.



[AWS_Portal]: "https://docs.thinkboxsoftware.com/products/deadline/10.0/1_User%20Manual/manual/aws-portal-setup-overview.html#aws-portal-setup-components-overview-ref-label"
[AWS_checklist]: "https://docs.thinkboxsoftware.com/products/deadline/10.0/1_User%20Manual/manual/aws-portal-setup-checklist.html"
[UBL_Setup]: "https://docs.thinkboxsoftware.com/products/deadline/10.0/1_User%20Manual/manual/aws-portal/licensing-setup.html#aws-portal-licensing-setup-ref-label"
[deadline_limits]: https://docs.thinkboxsoftware.com/products/deadline/10.0/1_User%20Manual/manual/aws-portal/configure-limits.html#aws-portal-configure-limits-ref-label
[deadline_to_rcs]: https://docs.thinkboxsoftware.com/products/deadline/10.0/1_User%20Manual/manual/aws-portal-configuring.html#specify-your-remote-connection-server
